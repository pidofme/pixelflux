// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Python-free FFmpeg media primitives extracted from Pixelflux.
//!
//! The crate intentionally exposes owned byte buffers and configuration values
//! instead of Pixelflux's Python, Wayland, or transport types. Product-specific
//! policy belongs in the embedding adapter.

mod decoder;
mod vaapi;

pub use decoder::{decode_h264_luma, DecodedFrame, DecoderError, H264StreamDecoder};
pub use vaapi::{
    DrmPrimeFrame, DrmPrimePlane, EncodedPacket, H264Gop, HostPixelFormat, VaapiError,
    VaapiErrorKind, VaapiHostConfiguration, VaapiHostEncoder,
};
