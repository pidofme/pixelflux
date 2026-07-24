// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Software H.264 decode validation for independent and inter-frame access units.

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

/// Stateful software H.264 decoder used to validate a no-B-frame reference chain.
pub struct H264StreamDecoder {
    context: *mut ff::AVCodecContext,
    packet: *mut ff::AVPacket,
    frame: *mut ff::AVFrame,
}

// The decoder is moved to, and used exclusively by, one validation thread.
unsafe impl Send for H264StreamDecoder {}

impl H264StreamDecoder {
    /// Opens one software H.264 decoder context.
    pub fn open() -> Result<Self, DecoderError> {
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
            Ok(Self {
                context,
                packet,
                frame,
            })
        }
    }

    /// Decodes the next Annex-B access unit while preserving prior references.
    pub fn decode(&mut self, annex_b: &[u8]) -> Result<DecodedFrame, DecoderError> {
        self.decode_inner(annex_b, false)
    }

    fn decode_inner(
        &mut self,
        annex_b: &[u8],
        flush_on_eagain: bool,
    ) -> Result<DecodedFrame, DecoderError> {
        validate_access_unit(annex_b)?;
        let packet_size = i32::try_from(annex_b.len()).map_err(|_| {
            DecoderError::new(
                "validate H.264 access unit",
                "payload length does not fit FFmpeg",
            )
        })?;

        unsafe {
            ff::av_packet_unref(self.packet);
            let packet_status = ff::av_new_packet(self.packet, packet_size);
            if packet_status < 0 {
                return Err(DecoderError::ffmpeg(
                    "allocate decoder packet",
                    packet_status,
                ));
            }
            ptr::copy_nonoverlapping(annex_b.as_ptr(), (*self.packet).data, annex_b.len());
            let send_status = ff::avcodec_send_packet(self.context, self.packet);
            ff::av_packet_unref(self.packet);
            if send_status < 0 {
                return Err(DecoderError::ffmpeg(
                    "submit H.264 access unit",
                    send_status,
                ));
            }

            ff::av_frame_unref(self.frame);
            let mut receive_status = ff::avcodec_receive_frame(self.context, self.frame);
            if receive_status == ff::AVERROR(libc::EAGAIN) && flush_on_eagain {
                let flush_status = ff::avcodec_send_packet(self.context, ptr::null());
                if flush_status < 0 {
                    return Err(DecoderError::ffmpeg(
                        "flush software H.264 decoder",
                        flush_status,
                    ));
                }
                receive_status = ff::avcodec_receive_frame(self.context, self.frame);
            }
            if receive_status == ff::AVERROR(libc::EAGAIN) {
                return Err(DecoderError::new(
                    "receive decoded H.264 frame",
                    "decoder produced no frame for the submitted access unit",
                ));
            }
            if receive_status < 0 {
                return Err(DecoderError::ffmpeg(
                    "receive decoded H.264 frame",
                    receive_status,
                ));
            }

            decoded_luma(self.frame)
        }
    }
}

impl Drop for H264StreamDecoder {
    fn drop(&mut self) {
        unsafe {
            cleanup(&mut self.context, &mut self.packet, &mut self.frame);
        }
    }
}

/// Decodes one independently decodable Annex-B H.264 access unit with a fresh context.
pub fn decode_h264_luma(annex_b: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let mut decoder = H264StreamDecoder::open()?;
    decoder.decode_inner(annex_b, true)
}

fn validate_access_unit(annex_b: &[u8]) -> Result<(), DecoderError> {
    if annex_b.is_empty() || annex_b.len() > MAX_ENCODED_BYTES {
        return Err(DecoderError::new(
            "validate H.264 access unit",
            "payload is empty or exceeds the decoder bound",
        ));
    }
    Ok(())
}

unsafe fn decoded_luma(frame: *mut ff::AVFrame) -> Result<DecodedFrame, DecoderError> {
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
    Ok(DecodedFrame {
        width: u32::try_from(width).expect("positive i32 width fits u32"),
        height: u32::try_from(height).expect("positive i32 height fits u32"),
        luma_sum,
        luma_samples,
    })
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

    #[test]
    fn stream_decoder_rejects_an_empty_access_unit() {
        let mut decoder = H264StreamDecoder::open().expect("software decoder");
        assert!(decoder.decode(&[]).is_err());
    }
}
