// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Host-frame and DRM-PRIME VA-API encoding extracted from Pixelflux 2.0.0.
//!
//! Python, Wayland, stripe framing, and recording are removed. CPU capture uses
//! `hwupload,scale_vaapi=format=nv12`; same-device DMABUF input uses an owned
//! `AVDRMFrameDescriptor` with `hwmap,scale_vaapi=format=nv12`.

use std::ffi::{c_int, c_void, CStr, CString};
use std::fmt;
use std::mem;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::os::unix::ffi::OsStrExt as _;
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ffmpeg_sys_next as ff;

const ENCODER_SURFACE_POOL_SIZE: i32 = 20;
const MAX_REFRESH_HZ: u32 = 1_000;
const MAX_RAW_FRAME_BYTES: usize = 512 * 1024 * 1024;
const AV_DRM_MAX_PLANES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMObjectDescriptor {
    fd: c_int,
    size: usize,
    format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMPlaneDescriptor {
    object_index: c_int,
    offset: isize,
    pitch: isize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMLayerDescriptor {
    format: u32,
    nb_planes: c_int,
    planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMFrameDescriptor {
    nb_objects: c_int,
    objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    nb_layers: c_int,
    layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

struct DmabufResources {
    fds: Vec<c_int>,
    released: Arc<AtomicBool>,
}

unsafe fn discard_unwrapped_drm_frame(
    descriptor: *mut AVDRMFrameDescriptor,
    resources: &mut DmabufResources,
) {
    for fd in resources.fds.drain(..) {
        libc::close(fd);
    }
    ff::av_free(descriptor.cast::<c_void>());
}

unsafe extern "C" fn release_drm_frame(opaque: *mut c_void, data: *mut u8) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let resources = Box::from_raw(opaque.cast::<DmabufResources>());
        for fd in resources.fds {
            libc::close(fd);
        }
        if !data.is_null() {
            ff::av_free(data.cast::<c_void>());
        }
        resources.released.store(true, Ordering::Release);
    }));
}

/// One borrowed plane in an owned DRM-PRIME frame description.
#[derive(Clone, Copy, Debug)]
pub struct DrmPrimePlane<'a> {
    pub fd: BorrowedFd<'a>,
    pub offset: u32,
    pub stride: u32,
}

/// Borrowed DRM-PRIME metadata whose file descriptors are duplicated into FFmpeg ownership.
#[derive(Clone, Copy, Debug)]
pub struct DrmPrimeFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub planes: &'a [DrmPrimePlane<'a>],
}

impl DrmPrimeFrame<'_> {
    fn validate(self, width: i32, height: i32) -> Result<(), VaapiError> {
        if self.width != u32::try_from(width).unwrap_or_default()
            || self.height != u32::try_from(height).unwrap_or_default()
            || self.fourcc == 0
            || self.planes.is_empty()
            || self.planes.len() > AV_DRM_MAX_PLANES
            || self.planes.iter().any(|plane| plane.stride == 0)
        {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate DRM-PRIME frame",
                "dimensions, format, plane count, or stride are invalid",
            ));
        }
        Ok(())
    }
}

/// CPU pixel layouts accepted by the host-frame uploader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostPixelFormat {
    Rgba,
    Bgra,
}

impl HostPixelFormat {
    const fn ffmpeg(self) -> ff::AVPixelFormat {
        match self {
            Self::Rgba => ff::AVPixelFormat::AV_PIX_FMT_RGBA,
            Self::Bgra => ff::AVPixelFormat::AV_PIX_FMT_BGRA,
        }
    }
}

/// H.264 reference policy for the low-latency VA-API encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H264Gop {
    /// Every access unit is an independently decodable IDR diagnostic frame.
    AllIntra,
    /// Emit P-frames between requested or periodic IDRs, with no B-frames.
    LowLatency { keyframe_interval_frames: u32 },
}

impl H264Gop {
    fn validate(self) -> Result<(), VaapiError> {
        match self {
            Self::AllIntra => Ok(()),
            Self::LowLatency {
                keyframe_interval_frames,
            } if keyframe_interval_frames > 0 => Ok(()),
            Self::LowLatency { .. } => Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate GOP configuration",
                "keyframe interval must contain at least one frame",
            )),
        }
    }

    const fn forces_every_frame(self) -> bool {
        matches!(self, Self::AllIntra)
    }

    const fn keyframe_interval_frames(self) -> u32 {
        match self {
            Self::AllIntra => 1,
            Self::LowLatency {
                keyframe_interval_frames,
            } => keyframe_interval_frames,
        }
    }
}

/// Immutable configuration for one VA-API encoder context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaapiHostConfiguration {
    pub device: PathBuf,
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
    pub bitrate_bps: u32,
    pub pixel_format: HostPixelFormat,
    pub gop: H264Gop,
}

impl VaapiHostConfiguration {
    fn validate(&self) -> Result<(i32, i32, i32), VaapiError> {
        self.gop.validate()?;
        if self.device.as_os_str().is_empty()
            || self.width == 0
            || self.height == 0
            || self.width & 1 != 0
            || self.height & 1 != 0
            || !(1..=MAX_REFRESH_HZ).contains(&self.refresh_hz)
            || self.bitrate_bps == 0
        {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate configuration",
                "dimensions, refresh rate, bitrate, or render node are invalid",
            ));
        }
        let bytes = usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| {
                VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate configuration",
                    "frame byte size overflowed",
                )
            })?;
        if bytes > MAX_RAW_FRAME_BYTES {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate configuration",
                "frame exceeds the bounded host-frame size",
            ));
        }
        let width = i32::try_from(self.width).map_err(|_| {
            VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate configuration",
                "width does not fit FFmpeg",
            )
        })?;
        let height = i32::try_from(self.height).map_err(|_| {
            VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate configuration",
                "height does not fit FFmpeg",
            )
        })?;
        let refresh_hz = i32::try_from(self.refresh_hz).map_err(|_| {
            VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate configuration",
                "refresh rate does not fit FFmpeg",
            )
        })?;
        Ok((width, height, refresh_hz))
    }
}

/// One packet emitted for one submitted host frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedPacket {
    pub annex_b: Vec<u8>,
    pub keyframe: bool,
    pub pts: i64,
    pub input_released: bool,
}

/// Stable failure classes exposed to embedding adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaapiErrorKind {
    InvalidConfiguration,
    Unavailable,
    EncodeFailed,
    InvalidOutput,
}

/// An operator-facing native FFmpeg/VA-API failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaapiError {
    kind: VaapiErrorKind,
    operation: &'static str,
    detail: String,
}

impl VaapiError {
    fn new(kind: VaapiErrorKind, operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            operation,
            detail: detail.into(),
        }
    }

    fn ffmpeg(kind: VaapiErrorKind, operation: &'static str, code: i32) -> Self {
        Self::new(kind, operation, ffmpeg_error_string(code))
    }

    #[must_use]
    pub const fn kind(&self) -> VaapiErrorKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for VaapiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.operation, self.detail)
    }
}

impl std::error::Error for VaapiError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputMode {
    Host,
    DrmPrime,
}

/// In-process FFmpeg `h264_vaapi` encoder for owned CPU or DRM-PRIME frames.
pub struct VaapiHostEncoder {
    encoder_ctx: *mut ff::AVCodecContext,
    hw_device_ctx: *mut ff::AVBufferRef,
    drm_device_ctx: *mut ff::AVBufferRef,
    drm_frames_ctx: *mut ff::AVBufferRef,
    enc_frames_ctx: *mut ff::AVBufferRef,
    filter_graph: *mut ff::AVFilterGraph,
    buffersrc_ctx: *mut ff::AVFilterContext,
    buffersink_ctx: *mut ff::AVFilterContext,
    video_frame: *mut ff::AVFrame,
    filtered_frame: *mut ff::AVFrame,
    packet: *mut ff::AVPacket,
    width: i32,
    height: i32,
    refresh_hz: i32,
    pixel_format: HostPixelFormat,
    input_mode: InputMode,
    gop: H264Gop,
    frames_since_keyframe: u32,
}

// FFmpeg contexts are used only by the owning encoder thread.
unsafe impl Send for VaapiHostEncoder {}

impl VaapiHostEncoder {
    /// Opens the render node, derives a VA device, creates the NV12 surface
    /// pool, and preflights the host upload/filter/encoder path.
    pub fn open(configuration: &VaapiHostConfiguration) -> Result<Self, VaapiError> {
        Self::open_impl(configuration, InputMode::Host)
    }

    /// Opens the same VA-API encoder with a DRM-PRIME `hwmap` input graph.
    pub fn open_dmabuf(configuration: &VaapiHostConfiguration) -> Result<Self, VaapiError> {
        Self::open_impl(configuration, InputMode::DrmPrime)
    }

    fn open_impl(
        configuration: &VaapiHostConfiguration,
        input_mode: InputMode,
    ) -> Result<Self, VaapiError> {
        let (width, height, refresh_hz) = configuration.validate()?;
        let device = CString::new(configuration.device.as_os_str().as_bytes()).map_err(|_| {
            VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate render node",
                "render node contains an interior NUL byte",
            )
        })?;
        let mut encoder = Self::empty(
            width,
            height,
            refresh_hz,
            configuration.pixel_format,
            input_mode,
            configuration.gop,
        );

        unsafe {
            let status = ff::av_hwdevice_ctx_create(
                &mut encoder.drm_device_ctx,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM,
                device.as_ptr(),
                ptr::null_mut(),
                0,
            );
            check(
                status,
                VaapiErrorKind::Unavailable,
                "create DRM render-node device",
            )?;

            let status = ff::av_hwdevice_ctx_create_derived(
                &mut encoder.hw_device_ctx,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                encoder.drm_device_ctx,
                0,
            );
            check(status, VaapiErrorKind::Unavailable, "derive VA-API device")?;

            if input_mode == InputMode::DrmPrime {
                encoder.drm_frames_ctx = ff::av_hwframe_ctx_alloc(encoder.drm_device_ctx);
                if encoder.drm_frames_ctx.is_null() {
                    return Err(VaapiError::new(
                        VaapiErrorKind::Unavailable,
                        "allocate DRM-PRIME frames context",
                        "FFmpeg returned a null frames context",
                    ));
                }
                let frames = (*encoder.drm_frames_ctx).data as *mut ff::AVHWFramesContext;
                if frames.is_null() {
                    return Err(VaapiError::new(
                        VaapiErrorKind::Unavailable,
                        "allocate DRM-PRIME frames context",
                        "FFmpeg returned an empty frames context",
                    ));
                }
                (*frames).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
                (*frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGRA;
                (*frames).width = width;
                (*frames).height = height;
                (*frames).initial_pool_size = 0;
                check(
                    ff::av_hwframe_ctx_init(encoder.drm_frames_ctx),
                    VaapiErrorKind::Unavailable,
                    "initialize DRM-PRIME frames context",
                )?;
            }

            encoder.enc_frames_ctx = ff::av_hwframe_ctx_alloc(encoder.hw_device_ctx);
            if encoder.enc_frames_ctx.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "allocate VA-API surface pool",
                    "FFmpeg returned a null frames context",
                ));
            }
            let frames = (*encoder.enc_frames_ctx).data as *mut ff::AVHWFramesContext;
            if frames.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "allocate VA-API surface pool",
                    "FFmpeg returned an empty frames context",
                ));
            }
            (*frames).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
            (*frames).width = align(width, 16)?;
            (*frames).height = align(height, 32)?;
            (*frames).initial_pool_size = ENCODER_SURFACE_POOL_SIZE;
            check(
                ff::av_hwframe_ctx_init(encoder.enc_frames_ctx),
                VaapiErrorKind::Unavailable,
                "initialize VA-API surface pool",
            )?;

            let codec_name = CString::new("h264_vaapi").expect("static codec name");
            let codec = ff::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "find VA-API H.264 encoder",
                    "system FFmpeg does not expose h264_vaapi",
                ));
            }
            encoder.encoder_ctx = ff::avcodec_alloc_context3(codec);
            if encoder.encoder_ctx.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "allocate VA-API encoder",
                    "FFmpeg returned a null codec context",
                ));
            }
            (*encoder.encoder_ctx).width = width;
            (*encoder.encoder_ctx).height = height;
            (*encoder.encoder_ctx).time_base = ff::AVRational {
                num: 1,
                den: refresh_hz,
            };
            (*encoder.encoder_ctx).framerate = ff::AVRational {
                num: refresh_hz,
                den: 1,
            };
            (*encoder.encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*encoder.encoder_ctx).hw_device_ctx = ff::av_buffer_ref(encoder.hw_device_ctx);
            (*encoder.encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(encoder.enc_frames_ctx);
            (*encoder.encoder_ctx).max_b_frames = 0;
            (*encoder.encoder_ctx).gop_size = if configuration.gop.forces_every_frame() {
                1
            } else {
                i32::MAX
            };
            (*encoder.encoder_ctx).slices = 4;
            (*encoder.encoder_ctx).compression_level = 6;
            (*encoder.encoder_ctx).flags |= ff::AV_CODEC_FLAG_LOW_DELAY as i32;
            (*encoder.encoder_ctx).bit_rate = i64::from(configuration.bitrate_bps);

            let mut options: *mut ff::AVDictionary = ptr::null_mut();
            // Preserve the validated all-IDR helper's driver-selected rate
            // control for 5R-C parity. Pixelflux CBR/CQP policy returns with
            // the production inter-frame work in 5R-D.
            let option_result = (|| {
                set_dictionary(&mut options, "aud", "0")?;
                set_dictionary(&mut options, "bf", "0")?;
                set_dictionary(&mut options, "profile", "high")?;
                set_dictionary(&mut options, "level", "4.1")?;
                if configuration.gop.forces_every_frame() {
                    set_dictionary(&mut options, "g", "1")?;
                    set_dictionary(&mut options, "idr_interval", "1")?;
                }
                set_dictionary(&mut options, "async_depth", "1")
            })();
            if let Err(error) = option_result {
                ff::av_dict_free(&mut options);
                return Err(error);
            }
            let status = ff::avcodec_open2(encoder.encoder_ctx, codec, &mut options);
            ff::av_dict_free(&mut options);
            check(
                status,
                VaapiErrorKind::Unavailable,
                "open VA-API H.264 encoder",
            )?;

            encoder.filter_graph = ff::avfilter_graph_alloc();
            if encoder.filter_graph.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "allocate VA-API filter graph",
                    "FFmpeg returned a null filter graph",
                ));
            }
            encoder.build_filter_graph()?;

            encoder.video_frame = ff::av_frame_alloc();
            encoder.filtered_frame = ff::av_frame_alloc();
            encoder.packet = ff::av_packet_alloc();
            if encoder.video_frame.is_null()
                || encoder.filtered_frame.is_null()
                || encoder.packet.is_null()
            {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "allocate reusable FFmpeg media objects",
                    "FFmpeg returned a null frame or packet",
                ));
            }

            let mut preflight = ff::av_frame_alloc();
            if preflight.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "preflight VA-API surface pool",
                    "FFmpeg returned a null frame",
                ));
            }
            let status = ff::av_hwframe_get_buffer(encoder.enc_frames_ctx, preflight, 0);
            ff::av_frame_free(&mut preflight);
            check(
                status,
                VaapiErrorKind::Unavailable,
                "preflight VA-API surface pool",
            )?;
        }

        Ok(encoder)
    }

    unsafe fn build_filter_graph(&mut self) -> Result<(), VaapiError> {
        let source_name = CString::new("buffer").expect("static filter name");
        let sink_name = CString::new("buffersink").expect("static filter name");
        let source_filter = ff::avfilter_get_by_name(source_name.as_ptr());
        let sink_filter = ff::avfilter_get_by_name(sink_name.as_ptr());
        if source_filter.is_null() || sink_filter.is_null() {
            return Err(VaapiError::new(
                VaapiErrorKind::Unavailable,
                "find FFmpeg buffer filters",
                "system FFmpeg does not expose buffer and buffersink",
            ));
        }

        let source_instance = CString::new("in").expect("static instance name");
        self.buffersrc_ctx = ff::avfilter_graph_alloc_filter(
            self.filter_graph,
            source_filter,
            source_instance.as_ptr(),
        );
        if self.buffersrc_ctx.is_null() {
            return Err(VaapiError::new(
                VaapiErrorKind::Unavailable,
                "allocate host-frame buffer source",
                "FFmpeg returned a null filter context",
            ));
        }

        let parameters = ff::av_buffersrc_parameters_alloc();
        if parameters.is_null() {
            return Err(VaapiError::new(
                VaapiErrorKind::Unavailable,
                "allocate host-frame source parameters",
                "FFmpeg returned a null parameter block",
            ));
        }
        (*parameters).format = match self.input_mode {
            InputMode::Host => self.pixel_format.ffmpeg() as i32,
            InputMode::DrmPrime => ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32,
        };
        if self.input_mode == InputMode::DrmPrime {
            (*parameters).hw_frames_ctx = ff::av_buffer_ref(self.drm_frames_ctx);
        }
        (*parameters).width = self.width;
        (*parameters).height = self.height;
        (*parameters).time_base = ff::AVRational {
            num: 1,
            den: self.refresh_hz,
        };
        let status = ff::av_buffersrc_parameters_set(self.buffersrc_ctx, parameters);
        if !(*parameters).hw_frames_ctx.is_null() {
            ff::av_buffer_unref(&mut (*parameters).hw_frames_ctx);
        }
        ff::av_free(parameters.cast::<c_void>());
        check(
            status,
            VaapiErrorKind::Unavailable,
            "configure host-frame buffer source",
        )?;

        let source_arguments = CString::new(format!(
            "video_size={}x{}:time_base=1/{}:pixel_aspect=1/1",
            self.width, self.height, self.refresh_hz
        ))
        .expect("numeric filter arguments");
        check(
            ff::avfilter_init_str(self.buffersrc_ctx, source_arguments.as_ptr()),
            VaapiErrorKind::Unavailable,
            "initialize host-frame buffer source",
        )?;

        let sink_instance = CString::new("out").expect("static instance name");
        check(
            ff::avfilter_graph_create_filter(
                &mut self.buffersink_ctx,
                sink_filter,
                sink_instance.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
                self.filter_graph,
            ),
            VaapiErrorKind::Unavailable,
            "initialize VA-API buffer sink",
        )?;

        let staging = match self.input_mode {
            InputMode::Host => "hwupload",
            InputMode::DrmPrime => "hwmap",
        };
        let description = CString::new(format!(
            "{staging},scale_vaapi=w={}:h={}:format=nv12:out_color_matrix=bt709:out_range=tv",
            self.width, self.height
        ))
        .expect("numeric filter description");
        let mut segment: *mut ff::AVFilterGraphSegment = ptr::null_mut();
        let mut inputs: *mut ff::AVFilterInOut = ptr::null_mut();
        let mut outputs: *mut ff::AVFilterInOut = ptr::null_mut();

        let result = (|| {
            check(
                ff::avfilter_graph_segment_parse(
                    self.filter_graph,
                    description.as_ptr(),
                    0,
                    &mut segment,
                ),
                VaapiErrorKind::Unavailable,
                "parse VA-API filter graph",
            )?;
            check(
                ff::avfilter_graph_segment_create_filters(segment, 0),
                VaapiErrorKind::Unavailable,
                "create VA-API filters",
            )?;
            for index in 0..(*self.filter_graph).nb_filters {
                let filter = *(*self.filter_graph).filters.add(index as usize);
                if !filter.is_null() && (*filter).hw_device_ctx.is_null() {
                    (*filter).hw_device_ctx = ff::av_buffer_ref(self.hw_device_ctx);
                }
            }
            check(
                ff::avfilter_graph_segment_apply(segment, 0, &mut inputs, &mut outputs),
                VaapiErrorKind::Unavailable,
                "initialize VA-API filters",
            )?;
            if inputs.is_null() || outputs.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::Unavailable,
                    "link VA-API filter graph",
                    "filter chain did not expose one input and one output",
                ));
            }
            check(
                ff::avfilter_link(
                    self.buffersrc_ctx,
                    0,
                    (*inputs).filter_ctx,
                    (*inputs).pad_idx as u32,
                ),
                VaapiErrorKind::Unavailable,
                "link VA-API input staging filter",
            )?;
            check(
                ff::avfilter_link(
                    (*outputs).filter_ctx,
                    (*outputs).pad_idx as u32,
                    self.buffersink_ctx,
                    0,
                ),
                VaapiErrorKind::Unavailable,
                "link VA-API conversion sink",
            )?;
            check(
                ff::avfilter_graph_config(self.filter_graph, ptr::null_mut()),
                VaapiErrorKind::Unavailable,
                "configure VA-API filter graph",
            )
        })();

        ff::avfilter_inout_free(&mut inputs);
        ff::avfilter_inout_free(&mut outputs);
        ff::avfilter_graph_segment_free(&mut segment);
        result
    }

    /// Copies one CPU frame into a reusable FFmpeg frame, uploads it to the VA
    /// device, converts it to NV12 on that device, and emits exactly one packet.
    pub fn encode(
        &mut self,
        pixels: &[u8],
        stride: usize,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<EncodedPacket, VaapiError> {
        if self.input_mode != InputMode::Host {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate host frame",
                "encoder was opened for DRM-PRIME input",
            ));
        }
        let minimum_stride = usize::try_from(self.width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| {
                VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate host frame",
                    "frame stride overflowed",
                )
            })?;
        let expected = stride
            .checked_mul(usize::try_from(self.height).map_err(|_| {
                VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate host frame",
                    "frame height does not fit memory",
                )
            })?)
            .ok_or_else(|| {
                VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate host frame",
                    "frame byte size overflowed",
                )
            })?;
        if stride < minimum_stride || expected > MAX_RAW_FRAME_BYTES || pixels.len() < expected {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate host frame",
                "frame bytes do not match dimensions and stride",
            ));
        }

        unsafe {
            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).format = self.pixel_format.ffmpeg() as i32;
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).pts = pts;
            check(
                ff::av_frame_get_buffer(self.video_frame, 32),
                VaapiErrorKind::EncodeFailed,
                "allocate host-frame buffer",
            )?;
            check(
                ff::av_frame_make_writable(self.video_frame),
                VaapiErrorKind::EncodeFailed,
                "make host-frame buffer writable",
            )?;

            let destination_stride =
                usize::try_from((*self.video_frame).linesize[0]).map_err(|_| {
                    VaapiError::new(
                        VaapiErrorKind::EncodeFailed,
                        "copy host frame",
                        "FFmpeg returned a negative destination stride",
                    )
                })?;
            let row_bytes = minimum_stride;
            for row in 0..usize::try_from(self.height).unwrap_or_default() {
                let source_offset = row.checked_mul(stride).ok_or_else(|| {
                    VaapiError::new(
                        VaapiErrorKind::InvalidConfiguration,
                        "copy host frame",
                        "source row offset overflowed",
                    )
                })?;
                let destination_offset = row.checked_mul(destination_stride).ok_or_else(|| {
                    VaapiError::new(
                        VaapiErrorKind::EncodeFailed,
                        "copy host frame",
                        "destination row offset overflowed",
                    )
                })?;
                ptr::copy_nonoverlapping(
                    pixels.as_ptr().add(source_offset),
                    (*self.video_frame).data[0].add(destination_offset),
                    row_bytes,
                );
            }

            check(
                ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame),
                VaapiErrorKind::EncodeFailed,
                "upload host frame",
            )?;
            self.receive_and_encode(pts, force_keyframe)
        }
    }

    /// Wraps one DRM-PRIME frame in an FFmpeg descriptor, maps it to VA-API,
    /// converts it to NV12 on-device, and emits exactly one packet.
    pub fn encode_dmabuf(
        &mut self,
        frame: DrmPrimeFrame<'_>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<EncodedPacket, VaapiError> {
        if self.input_mode != InputMode::DrmPrime {
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "validate DRM-PRIME frame",
                "encoder was opened for host input",
            ));
        }
        frame.validate(self.width, self.height)?;

        unsafe {
            let descriptor_size = mem::size_of::<AVDRMFrameDescriptor>();
            let descriptor = ff::av_mallocz(descriptor_size).cast::<AVDRMFrameDescriptor>();
            if descriptor.is_null() {
                return Err(VaapiError::new(
                    VaapiErrorKind::EncodeFailed,
                    "allocate DRM-PRIME descriptor",
                    "FFmpeg returned a null allocation",
                ));
            }

            let release_observed = Arc::new(AtomicBool::new(false));
            let mut resources = DmabufResources {
                fds: Vec::with_capacity(frame.planes.len()),
                released: Arc::clone(&release_observed),
            };
            let Ok(nb_objects) = c_int::try_from(frame.planes.len()) else {
                discard_unwrapped_drm_frame(descriptor, &mut resources);
                return Err(VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate DRM-PRIME frame",
                    "plane count does not fit FFmpeg",
                ));
            };
            (*descriptor).nb_objects = nb_objects;
            (*descriptor).nb_layers = 1;
            (*descriptor).layers[0].format = frame.fourcc;
            (*descriptor).layers[0].nb_planes = (*descriptor).nb_objects;

            let aligned_height = usize::try_from(align(self.height, 32)?).map_err(|_| {
                VaapiError::new(
                    VaapiErrorKind::InvalidConfiguration,
                    "validate DRM-PRIME frame",
                    "aligned height does not fit memory",
                )
            })?;
            for (index, plane) in frame.planes.iter().enumerate() {
                let fd = libc::dup(plane.fd.as_raw_fd());
                if fd < 0 {
                    discard_unwrapped_drm_frame(descriptor, &mut resources);
                    return Err(VaapiError::new(
                        VaapiErrorKind::EncodeFailed,
                        "duplicate DRM-PRIME file descriptor",
                        std::io::Error::last_os_error().to_string(),
                    ));
                }
                resources.fds.push(fd);
                let Some(object_size) = usize::try_from(plane.offset).ok().and_then(|offset| {
                    usize::try_from(plane.stride)
                        .ok()
                        .and_then(|stride| stride.checked_mul(aligned_height))
                        .and_then(|bytes| offset.checked_add(bytes))
                }) else {
                    discard_unwrapped_drm_frame(descriptor, &mut resources);
                    return Err(VaapiError::new(
                        VaapiErrorKind::InvalidConfiguration,
                        "validate DRM-PRIME frame",
                        "plane object size overflowed",
                    ));
                };
                (*descriptor).objects[index].fd = fd;
                (*descriptor).objects[index].size = object_size;
                (*descriptor).objects[index].format_modifier = frame.modifier;
                let Ok(object_index) = c_int::try_from(index) else {
                    discard_unwrapped_drm_frame(descriptor, &mut resources);
                    return Err(VaapiError::new(
                        VaapiErrorKind::InvalidConfiguration,
                        "validate DRM-PRIME frame",
                        "plane index does not fit FFmpeg",
                    ));
                };
                let Ok(offset) = isize::try_from(plane.offset) else {
                    discard_unwrapped_drm_frame(descriptor, &mut resources);
                    return Err(VaapiError::new(
                        VaapiErrorKind::InvalidConfiguration,
                        "validate DRM-PRIME frame",
                        "plane offset does not fit FFmpeg",
                    ));
                };
                let Ok(pitch) = isize::try_from(plane.stride) else {
                    discard_unwrapped_drm_frame(descriptor, &mut resources);
                    return Err(VaapiError::new(
                        VaapiErrorKind::InvalidConfiguration,
                        "validate DRM-PRIME frame",
                        "plane stride does not fit FFmpeg",
                    ));
                };
                (*descriptor).layers[0].planes[index].object_index = object_index;
                (*descriptor).layers[0].planes[index].offset = offset;
                (*descriptor).layers[0].planes[index].pitch = pitch;
            }

            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*self.video_frame).data[0] = descriptor.cast::<u8>();
            let opaque = Box::into_raw(Box::new(resources));
            let buffer = ff::av_buffer_create(
                descriptor.cast::<u8>(),
                descriptor_size,
                Some(release_drm_frame),
                opaque.cast::<c_void>(),
                0,
            );
            if buffer.is_null() {
                release_drm_frame(opaque.cast::<c_void>(), ptr::null_mut());
                ff::av_free(descriptor.cast::<c_void>());
                return Err(VaapiError::new(
                    VaapiErrorKind::EncodeFailed,
                    "wrap DRM-PRIME descriptor",
                    "FFmpeg returned a null buffer reference",
                ));
            }
            (*self.video_frame).buf[0] = buffer;
            (*self.video_frame).pts = pts;
            (*self.video_frame).hw_frames_ctx = ff::av_buffer_ref(self.drm_frames_ctx);
            if (*self.video_frame).hw_frames_ctx.is_null() {
                ff::av_frame_unref(self.video_frame);
                return Err(VaapiError::new(
                    VaapiErrorKind::EncodeFailed,
                    "wrap DRM-PRIME frame",
                    "FFmpeg could not retain the DRM frames context",
                ));
            }

            let status = ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame);
            if status < 0 {
                ff::av_frame_unref(self.video_frame);
                return Err(VaapiError::ffmpeg(
                    VaapiErrorKind::EncodeFailed,
                    "map DRM-PRIME frame",
                    status,
                ));
            }
            let encoded = self.receive_and_encode(pts, force_keyframe);
            ff::av_frame_unref(self.video_frame);
            if !release_observed.load(Ordering::Acquire) {
                ff::avfilter_graph_free(&mut self.filter_graph);
                self.buffersrc_ctx = ptr::null_mut();
                self.buffersink_ctx = ptr::null_mut();
                if !release_observed.load(Ordering::Acquire) {
                    return Err(VaapiError::new(
                        VaapiErrorKind::EncodeFailed,
                        "release DRM-PRIME frame",
                        "FFmpeg did not release the imported frame when its graph was destroyed",
                    ));
                }
                return Err(VaapiError::new(
                    VaapiErrorKind::EncodeFailed,
                    "release DRM-PRIME frame",
                    "FFmpeg retained the imported frame beyond one converted output",
                ));
            }
            encoded.map(|mut packet| {
                packet.input_released = true;
                packet
            })
        }
    }

    unsafe fn receive_and_encode(
        &mut self,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<EncodedPacket, VaapiError> {
        ff::av_frame_unref(self.filtered_frame);
        check(
            ff::av_buffersink_get_frame(self.buffersink_ctx, self.filtered_frame),
            VaapiErrorKind::EncodeFailed,
            "receive converted VA-API frame",
        )?;
        let keyframe_due = self.gop.forces_every_frame()
            || self.frames_since_keyframe >= self.gop.keyframe_interval_frames();
        if force_keyframe || keyframe_due {
            (*self.filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
        } else {
            (*self.filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_NONE;
        }
        (*self.filtered_frame).pts = pts;
        let send_status = ff::avcodec_send_frame(self.encoder_ctx, self.filtered_frame);
        ff::av_frame_unref(self.filtered_frame);
        check(
            send_status,
            VaapiErrorKind::EncodeFailed,
            "submit VA-API frame",
        )?;

        ff::av_packet_unref(self.packet);
        let receive_status = ff::avcodec_receive_packet(self.encoder_ctx, self.packet);
        if receive_status == ff::AVERROR(libc::EAGAIN) {
            return Err(VaapiError::new(
                VaapiErrorKind::EncodeFailed,
                "receive VA-API packet",
                "low-latency encoder produced no packet for the submitted frame",
            ));
        }
        check(
            receive_status,
            VaapiErrorKind::EncodeFailed,
            "receive VA-API packet",
        )?;
        if (*self.packet).pts != pts {
            ff::av_packet_unref(self.packet);
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidOutput,
                "validate VA-API packet",
                "packet PTS does not match the submitted frame",
            ));
        }
        let size = usize::try_from((*self.packet).size).map_err(|_| {
            VaapiError::new(
                VaapiErrorKind::InvalidOutput,
                "validate VA-API packet",
                "packet size is negative",
            )
        })?;
        if size == 0 || (*self.packet).data.is_null() {
            ff::av_packet_unref(self.packet);
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidOutput,
                "validate VA-API packet",
                "encoder returned an empty packet",
            ));
        }
        let packet = EncodedPacket {
            annex_b: slice::from_raw_parts((*self.packet).data, size).to_vec(),
            keyframe: ((*self.packet).flags & ff::AV_PKT_FLAG_KEY) != 0,
            pts: (*self.packet).pts,
            input_released: self.input_mode == InputMode::Host,
        };
        if packet.keyframe {
            self.frames_since_keyframe = 0;
        } else {
            self.frames_since_keyframe = self.frames_since_keyframe.saturating_add(1);
        }
        ff::av_packet_unref(self.packet);

        let extra_status = ff::avcodec_receive_packet(self.encoder_ctx, self.packet);
        if extra_status == 0 {
            ff::av_packet_unref(self.packet);
            return Err(VaapiError::new(
                VaapiErrorKind::InvalidOutput,
                "validate VA-API packet count",
                "one submitted frame produced more than one packet",
            ));
        }
        if extra_status != ff::AVERROR(libc::EAGAIN) && extra_status != ff::AVERROR_EOF {
            return Err(VaapiError::ffmpeg(
                VaapiErrorKind::EncodeFailed,
                "drain VA-API encoder",
                extra_status,
            ));
        }
        Ok(packet)
    }

    fn empty(
        width: i32,
        height: i32,
        refresh_hz: i32,
        pixel_format: HostPixelFormat,
        input_mode: InputMode,
        gop: H264Gop,
    ) -> Self {
        Self {
            encoder_ctx: ptr::null_mut(),
            hw_device_ctx: ptr::null_mut(),
            drm_device_ctx: ptr::null_mut(),
            drm_frames_ctx: ptr::null_mut(),
            enc_frames_ctx: ptr::null_mut(),
            filter_graph: ptr::null_mut(),
            buffersrc_ctx: ptr::null_mut(),
            buffersink_ctx: ptr::null_mut(),
            video_frame: ptr::null_mut(),
            filtered_frame: ptr::null_mut(),
            packet: ptr::null_mut(),
            width,
            height,
            refresh_hz,
            pixel_format,
            input_mode,
            gop,
            frames_since_keyframe: gop.keyframe_interval_frames(),
        }
    }
}

impl Drop for VaapiHostEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.packet.is_null() {
                ff::av_packet_free(&mut self.packet);
            }
            if !self.filtered_frame.is_null() {
                ff::av_frame_free(&mut self.filtered_frame);
            }
            if !self.video_frame.is_null() {
                ff::av_frame_free(&mut self.video_frame);
            }
            if !self.filter_graph.is_null() {
                ff::avfilter_graph_free(&mut self.filter_graph);
            }
            if !self.encoder_ctx.is_null() {
                ff::avcodec_free_context(&mut self.encoder_ctx);
            }
            if !self.enc_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.enc_frames_ctx);
            }
            if !self.drm_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_frames_ctx);
            }
            if !self.hw_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.hw_device_ctx);
            }
            if !self.drm_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_device_ctx);
            }
        }
    }
}

unsafe fn set_dictionary(
    dictionary: &mut *mut ff::AVDictionary,
    key: &str,
    value: &str,
) -> Result<(), VaapiError> {
    let key = CString::new(key).expect("static option key");
    let value = CString::new(value).map_err(|_| {
        VaapiError::new(
            VaapiErrorKind::InvalidConfiguration,
            "configure VA-API encoder",
            "option value contains an interior NUL byte",
        )
    })?;
    check(
        ff::av_dict_set(dictionary, key.as_ptr(), value.as_ptr(), 0),
        VaapiErrorKind::Unavailable,
        "configure VA-API encoder",
    )
}

fn align(value: i32, alignment: i32) -> Result<i32, VaapiError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| {
            VaapiError::new(
                VaapiErrorKind::InvalidConfiguration,
                "align VA-API surface dimensions",
                "surface dimension overflowed",
            )
        })
}

fn check(status: i32, kind: VaapiErrorKind, operation: &'static str) -> Result<(), VaapiError> {
    if status < 0 {
        Err(VaapiError::ffmpeg(kind, operation, status))
    } else {
        Ok(())
    }
}

fn ffmpeg_error_string(status: i32) -> String {
    let mut buffer = [0_i8; 256];
    unsafe {
        if ff::av_strerror(status, buffer.as_mut_ptr(), buffer.len()) == 0 {
            CStr::from_ptr(buffer.as_ptr())
                .to_string_lossy()
                .into_owned()
        } else {
            format!("FFmpeg error {status}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::os::fd::AsFd as _;

    use super::*;

    fn configuration() -> VaapiHostConfiguration {
        VaapiHostConfiguration {
            device: PathBuf::from("<gpu-render-node>"),
            width: 320,
            height: 240,
            refresh_hz: 30,
            bitrate_bps: 1_000_000,
            pixel_format: HostPixelFormat::Rgba,
            gop: H264Gop::AllIntra,
        }
    }

    #[test]
    fn dimensions_must_be_even_and_bounded() {
        assert_eq!(configuration().validate(), Ok((320, 240, 30)));
        let mut invalid = configuration();
        invalid.width = 319;
        assert_eq!(
            invalid.validate().expect_err("odd width must fail").kind(),
            VaapiErrorKind::InvalidConfiguration,
        );
    }

    #[test]
    fn surface_alignment_matches_vaapi_pool_contract() {
        assert_eq!(align(320, 16), Ok(320));
        assert_eq!(align(241, 32), Ok(256));
    }

    #[test]
    fn drm_prime_metadata_is_bounded_without_opening_ffmpeg() {
        let file = File::open("/dev/null").expect("test file descriptor");
        let planes = [DrmPrimePlane {
            fd: file.as_fd(),
            offset: 0,
            stride: 1_280,
        }];
        assert!(DrmPrimeFrame {
            width: 320,
            height: 240,
            fourcc: u32::from_le_bytes(*b"AR24"),
            modifier: 0,
            planes: &planes,
        }
        .validate(320, 240)
        .is_ok());
        assert!(DrmPrimeFrame {
            width: 320,
            height: 240,
            fourcc: u32::from_le_bytes(*b"AR24"),
            modifier: 0,
            planes: &[],
        }
        .validate(320, 240)
        .is_err());
    }

    #[test]
    fn production_gop_requires_a_bounded_nonzero_interval() {
        let mut invalid = configuration();
        invalid.gop = H264Gop::LowLatency {
            keyframe_interval_frames: 0,
        };
        assert_eq!(
            invalid
                .validate()
                .expect_err("zero GOP interval must fail")
                .kind(),
            VaapiErrorKind::InvalidConfiguration,
        );
        let mut production = configuration();
        production.gop = H264Gop::LowLatency {
            keyframe_interval_frames: 60,
        };
        assert_eq!(production.validate(), Ok((320, 240, 30)));
    }
}
