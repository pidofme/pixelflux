// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Fresh-context software H.264 decode used to verify hardware encoder output.

use std::ffi::CStr;
use std::fmt;
use std::ptr;
use std::slice;

use ffmpeg_sys_next as ff;

const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub luma_sum: u64,
    pub luma_samples: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderError {
    operation: &'static str,
    detail: String,
}

impl DecoderError {
    fn new(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            operation,
            detail: detail.into(),
        }
    }

    fn ffmpeg(operation: &'static str, status: i32) -> Self {
        Self::new(operation, ffmpeg_error_string(status))
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for DecoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.operation, self.detail)
    }
}

impl std::error::Error for DecoderError {}

/// Decodes one Annex-B H.264 access unit with a newly allocated software
/// decoder context and returns bounded luma statistics.
pub fn decode_h264_luma(annex_b: &[u8]) -> Result<DecodedFrame, DecoderError> {
    if annex_b.is_empty() || annex_b.len() > MAX_ENCODED_BYTES {
        return Err(DecoderError::new(
            "validate H.264 access unit",
            "payload is empty or exceeds the decoder bound",
        ));
    }
    let packet_size = i32::try_from(annex_b.len()).map_err(|_| {
        DecoderError::new(
            "validate H.264 access unit",
            "payload length does not fit FFmpeg",
        )
    })?;

    unsafe {
        let decoder = ff::avcodec_find_decoder(ff::AVCodecID::AV_CODEC_ID_H264);
        if decoder.is_null() {
            return Err(DecoderError::new(
                "find software H.264 decoder",
                "system FFmpeg does not expose an H.264 decoder",
            ));
        }
        let mut context = ff::avcodec_alloc_context3(decoder);
        let mut packet = ff::av_packet_alloc();
        let mut frame = ff::av_frame_alloc();
        if context.is_null() || packet.is_null() || frame.is_null() {
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(DecoderError::new(
                "allocate H.264 decoder",
                "FFmpeg returned a null context, packet, or frame",
            ));
        }
        (*context).thread_count = 1;
        let open_status = ff::avcodec_open2(context, decoder, ptr::null_mut());
        if open_status < 0 {
            let error = DecoderError::ffmpeg("open software H.264 decoder", open_status);
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(error);
        }
        let packet_status = ff::av_new_packet(packet, packet_size);
        if packet_status < 0 {
            let error = DecoderError::ffmpeg("allocate decoder packet", packet_status);
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(error);
        }
        ptr::copy_nonoverlapping(annex_b.as_ptr(), (*packet).data, annex_b.len());
        let send_status = ff::avcodec_send_packet(context, packet);
        if send_status < 0 {
            let error = DecoderError::ffmpeg("submit H.264 access unit", send_status);
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(error);
        }

        let mut receive_status = ff::avcodec_receive_frame(context, frame);
        if receive_status == ff::AVERROR(libc::EAGAIN) {
            let flush_status = ff::avcodec_send_packet(context, ptr::null());
            if flush_status < 0 {
                let error = DecoderError::ffmpeg("flush software H.264 decoder", flush_status);
                cleanup(&mut context, &mut packet, &mut frame);
                return Err(error);
            }
            receive_status = ff::avcodec_receive_frame(context, frame);
        }
        if receive_status < 0 {
            let error = DecoderError::ffmpeg("receive decoded H.264 frame", receive_status);
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(error);
        }

        let width = (*frame).width;
        let height = (*frame).height;
        let stride = (*frame).linesize[0];
        let supported_format = (*frame).format == ff::AVPixelFormat::AV_PIX_FMT_YUV420P as i32
            || (*frame).format == ff::AVPixelFormat::AV_PIX_FMT_YUVJ420P as i32
            || (*frame).format == ff::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
        if width <= 0
            || height <= 0
            || stride < width
            || (*frame).data[0].is_null()
            || !supported_format
        {
            cleanup(&mut context, &mut packet, &mut frame);
            return Err(DecoderError::new(
                "validate decoded H.264 frame",
                "decoder returned an unsupported or invalid luma plane",
            ));
        }

        let width_usize = usize::try_from(width).map_err(|_| {
            DecoderError::new(
                "validate decoded H.264 frame",
                "decoded width does not fit memory",
            )
        })?;
        let height_usize = usize::try_from(height).map_err(|_| {
            DecoderError::new(
                "validate decoded H.264 frame",
                "decoded height does not fit memory",
            )
        })?;
        let stride_usize = usize::try_from(stride).map_err(|_| {
            DecoderError::new(
                "validate decoded H.264 frame",
                "decoded stride does not fit memory",
            )
        })?;
        let mut luma_sum = 0_u64;
        for row in 0..height_usize {
            let row_pointer = (*frame).data[0].add(row * stride_usize);
            for value in slice::from_raw_parts(row_pointer, width_usize) {
                luma_sum = luma_sum.saturating_add(u64::from(*value));
            }
        }
        let luma_samples = u64::try_from(width_usize)
            .ok()
            .and_then(|width| {
                u64::try_from(height_usize)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| {
                DecoderError::new(
                    "validate decoded H.264 frame",
                    "decoded luma sample count overflowed",
                )
            })?;
        let decoded = DecodedFrame {
            width: u32::try_from(width).expect("positive i32 width fits u32"),
            height: u32::try_from(height).expect("positive i32 height fits u32"),
            luma_sum,
            luma_samples,
        };
        cleanup(&mut context, &mut packet, &mut frame);
        Ok(decoded)
    }
}

unsafe fn cleanup(
    context: &mut *mut ff::AVCodecContext,
    packet: &mut *mut ff::AVPacket,
    frame: &mut *mut ff::AVFrame,
) {
    ff::av_frame_free(frame);
    ff::av_packet_free(packet);
    ff::avcodec_free_context(context);
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
    use super::*;

    #[test]
    fn empty_access_unit_is_rejected_without_opening_a_decoder() {
        assert!(decode_h264_luma(&[]).is_err());
    }
}
