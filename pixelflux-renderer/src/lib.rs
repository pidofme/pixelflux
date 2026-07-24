// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Python-free GLES/GBM render target extracted from Pixelflux 2.0.0.
//!
//! The target keeps the GBM buffer object alive for exactly as long as its
//! exported DMABUF and GLES framebuffer are usable. Product/session policy,
//! fallback, capture cadence, and transport remain outside this crate.

use std::fs::File;
use std::path::Path;

use gbm::{BufferObject, BufferObjectFlags, Device as RawGbmDevice, Format as GbmFormat};
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufFlags};
use smithay::backend::allocator::gbm::GbmDevice;
use smithay::backend::allocator::{Fourcc, Modifier};
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::ImportEgl;
use smithay::reexports::wayland_server::DisplayHandle;
use thiserror::Error;

const MAX_DIMENSION: u32 = 8_192;
const TARGET_POOL_SIZE: usize = 2;
const _: () = assert!(TARGET_POOL_SIZE >= 2);

/// Classified GLES/GBM initialization and allocation failure.
#[derive(Debug, Error)]
pub enum RendererError {
    #[error("invalid GLES/GBM render-target configuration")]
    InvalidConfiguration,
    #[error("failed to open the DRM render node: {0}")]
    OpenDevice(#[source] std::io::Error),
    #[error("failed to initialize the GLES/GBM renderer at {0}")]
    Initialize(&'static str),
    #[error("failed to allocate the GLES/GBM render target")]
    Allocate,
    #[error("all GLES/GBM render targets are leased")]
    Busy,
}

struct RenderTarget {
    _buffer_object: BufferObject<()>,
    dmabuf: Dmabuf,
    leased: bool,
}

/// One GLES renderer and a bounded pool of ARGB8888 GBM targets exported as DMABUFs.
pub struct GlesGbmTarget {
    renderer: GlesRenderer,
    allocator: RawGbmDevice<File>,
    targets: Vec<RenderTarget>,
    render_node: DrmNode,
    width: u32,
    height: u32,
}

impl GlesGbmTarget {
    /// Opens the render node and creates EGL, GLES, GBM, and the first target.
    pub fn open(path: &Path, width: u32, height: u32) -> Result<Self, RendererError> {
        validate_dimensions(width, height)?;
        let file = File::options()
            .read(true)
            .write(true)
            .open(path)
            .map_err(RendererError::OpenDevice)?;
        let allocator_file = file.try_clone().map_err(RendererError::OpenDevice)?;
        let allocator = RawGbmDevice::new(allocator_file)
            .map_err(|_| RendererError::Initialize("GBM allocator"))?;
        let gbm = GbmDevice::new(file).map_err(|_| RendererError::Initialize("GBM device"))?;
        let egl = unsafe { EGLDisplay::new(gbm) }
            .map_err(|_| RendererError::Initialize("EGL display"))?;
        let context =
            EGLContext::new(&egl).map_err(|_| RendererError::Initialize("EGL context"))?;
        let renderer = unsafe { GlesRenderer::new(context) }
            .map_err(|_| RendererError::Initialize("GLES renderer"))?;
        let render_node =
            DrmNode::from_path(path).map_err(|_| RendererError::Initialize("DRM node"))?;
        let targets = allocate_targets(&allocator, &render_node, width, height)?;

        Ok(Self {
            renderer,
            allocator,
            targets,
            render_node,
            width,
            height,
        })
    }

    /// Binds EGL buffer import to the owning Wayland display.
    pub fn bind_wayland_display(&mut self, display: &DisplayHandle) -> Result<(), RendererError> {
        self.renderer
            .bind_wl_display(display)
            .map_err(|_| RendererError::Initialize("Wayland EGL binding"))
    }

    /// Leases one free target without blocking the compositor thread.
    #[must_use]
    pub fn acquire_target(&mut self) -> Option<usize> {
        let (index, target) = self
            .targets
            .iter_mut()
            .enumerate()
            .find(|(_, target)| !target.leased)?;
        target.leased = true;
        Some(index)
    }

    /// Returns disjoint mutable renderer and framebuffer handles for one leased render pass.
    pub fn render_parts(&mut self, index: usize) -> Option<(&mut GlesRenderer, &mut Dmabuf)> {
        let target = self.targets.get_mut(index)?;
        target
            .leased
            .then_some((&mut self.renderer, &mut target.dmabuf))
    }

    /// Returns one exported leased target while retaining the backing GBM object in `self`.
    #[must_use]
    pub fn dmabuf(&self, index: usize) -> Option<&Dmabuf> {
        self.targets
            .get(index)
            .filter(|target| target.leased)
            .map(|target| &target.dmabuf)
    }

    /// Releases one target after all cross-thread frame owners have dropped it.
    pub fn release_target(&mut self, index: usize) -> bool {
        let Some(target) = self.targets.get_mut(index) else {
            return false;
        };
        let was_leased = target.leased;
        target.leased = false;
        was_leased
    }

    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.targets.len()
    }

    #[must_use]
    pub const fn render_node(&self) -> DrmNode {
        self.render_node
    }

    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Allocates a complete replacement pool before releasing the current targets.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererError> {
        validate_dimensions(width, height)?;
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        if self.targets.iter().any(|target| target.leased) {
            return Err(RendererError::Busy);
        }
        let targets = allocate_targets(&self.allocator, &self.render_node, width, height)?;
        self.targets = targets;
        self.width = width;
        self.height = height;
        Ok(())
    }
}

fn allocate_targets(
    allocator: &RawGbmDevice<File>,
    render_node: &DrmNode,
    width: u32,
    height: u32,
) -> Result<Vec<RenderTarget>, RendererError> {
    (0..TARGET_POOL_SIZE)
        .map(|_| allocate_target(allocator, render_node, width, height))
        .collect()
}

fn allocate_target(
    allocator: &RawGbmDevice<File>,
    render_node: &DrmNode,
    width: u32,
    height: u32,
) -> Result<RenderTarget, RendererError> {
    let buffer_object = allocator
        .create_buffer_object(
            width,
            height,
            GbmFormat::Argb8888,
            BufferObjectFlags::RENDERING,
        )
        .map_err(|_| RendererError::Allocate)?;
    let fd = buffer_object.fd().map_err(|_| RendererError::Allocate)?;
    let modifier = Modifier::from(Into::<u64>::into(buffer_object.modifier()));
    let mut builder = Dmabuf::builder(
        (width as i32, height as i32),
        Fourcc::Argb8888,
        modifier,
        DmabufFlags::empty(),
    );
    if !builder.add_plane(fd, 0, 0, buffer_object.stride()) {
        return Err(RendererError::Allocate);
    }
    builder.set_node(*render_node);
    let dmabuf = builder.build().ok_or(RendererError::Allocate)?;
    Ok(RenderTarget {
        _buffer_object: buffer_object,
        dmabuf,
        leased: false,
    })
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), RendererError> {
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(RendererError::InvalidConfiguration);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_are_bounded_without_opening_a_device() {
        assert!(validate_dimensions(320, 240).is_ok());
        assert!(validate_dimensions(0, 240).is_err());
        assert!(validate_dimensions(320, 0).is_err());
        assert!(validate_dimensions(MAX_DIMENSION + 1, 240).is_err());
    }
}
