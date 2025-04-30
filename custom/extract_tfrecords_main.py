# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Produces tfrecord files similar to the YouTube-8M dataset.

It processes a CSV file containing lines like "<video_file>,<labels>", where
<video_file> must be a path to a video, and <labels> must be an integer list
joined with semi-colon ";". It processes all videos and outputs a TFRecord
file at --output_tfrecords_file.

It assumes that you have OpenCV installed and properly linked with ffmpeg (i.e.
cv2.VideoCapture().open('/path/to/video') should return True).

The binary only processes the video stream (images) and not the audio stream.
"""

import argparse
import csv
import os
import sys

import cv2
import feature_extractor
import numpy as np
import tensorflow as tf
# Para usar GraphDef y Session bajo TF2
tf.compat.v1.disable_eager_execution()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera TFRecord al estilo YouTube-8M desde vídeo local"
    )
    parser.add_argument(
        '--input_videos_csv', required=True,
        help='CSV con líneas "<ruta_video>,<etiquetas>"'
    )
    parser.add_argument(
        '--output_tfrecords_file', required=True,
        help='Ruta al TFRecord de salida'
    )
    parser.add_argument(
        '--model_dir',
        default=os.path.join(os.getenv('HOME', ''), 'yt8m'),
        help='Directorio de modelos Inception+PCA (descarga automática si falta)'
    )
    parser.add_argument(
        '--frames_per_second', type=float, default=1.0,
        help='FPS para extraer frames (frame-level features)'
    )
    parser.add_argument(
        '--skip_frame_level_features', action='store_true',
        help='Si se setea, omite features frame-level y solo escribe video-level features'
    )
    parser.add_argument(
        '--labels_feature_key', default='labels',
        help='Clave para escribir las etiquetas en el contexto del SequenceExample'
    )
    parser.add_argument(
        '--image_feature_key', default='rgb',
        help='Clave para escribir los features de imagen en las feature_lists'
    )
    parser.add_argument(
        '--video_file_feature_key', default='id',
        help='Clave para escribir la ruta del vídeo en el contexto (solo debugging)'
    )
    parser.add_argument(
        '--insert_zero_audio_features', action='store_true',
        help='Si se setea, inserta features de audio a cero de 128-D por frame'
    )
    return parser.parse_args()


# Propiedad de timestamp en OpenCV
# En OpenCV ≥3.x: cv2.CAP_PROP_POS_MSEC; en versiones antiguas podría ser cv2.cv.CV_CAP_PROP_POS_MSEC
if hasattr(cv2, 'CAP_PROP_POS_MSEC'):
    CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC
else:
    CAP_PROP_POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC  # type: ignore


def frame_iterator(filename, every_ms=1000, max_num_frames=300):
    """
    Itera sobre los frames de un vídeo a una frecuencia dada.

    Args:
      filename: Path al archivo de vídeo.
      every_ms: Milisegundos entre cada frame extraído.
      max_num_frames: Número máximo de frames a procesar.

    Yields:
      Frame en RGB con shape (alto, ancho, canales).
    """
    cap = cv2.VideoCapture()
    if not cap.open(filename):
        print(f"Error: Cannot open video file {filename}", file=sys.stderr)
        return
    last_ts = -every_ms
    num_retrieved = 0

    while num_retrieved < max_num_frames:
        # Avanzar hasta el siguiente timestamp
        while cap.get(CAP_PROP_POS_MSEC) < last_ts + every_ms:
            if not cap.read()[0]:
                cap.release()
                return
        last_ts = cap.get(CAP_PROP_POS_MSEC)
        has_frame, frame = cap.read()
        if not has_frame:
            break
        # Convertir BGR→RGB
        yield frame[:, :, ::-1]
        num_retrieved += 1

    cap.release()


def _int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
    # Convierte lista de ints a bytes
    return bytes(int_array) if sys.version_info[0] >= 3 else ''.join(map(chr, int_array))


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    """
    Cuantiza un array float32 a bytes (0-255), como en YouTube-8M.
    """
    assert features.dtype == np.float32
    assert features.ndim == 1
    clipped = np.clip(features, min_quantized_value, max_quantized_value)
    scale = 255.0 / (max_quantized_value - min_quantized_value)
    ints = np.round((clipped - min_quantized_value) * scale).astype(np.uint8)
    return _make_bytes(ints.tolist())


def main(args):
    extractor = feature_extractor.YouTube8MFeatureExtractor(args.model_dir)
    writer = tf.io.TFRecordWriter(args.output_tfrecords_file)
    total_written = 0
    total_error = 0

    with open(args.input_videos_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for video_file, labels in reader:
            rgb_features = []
            sum_rgb = None

            # Extraer frames y features
            for rgb in frame_iterator(
                video_file,
                every_ms=1000.0 / args.frames_per_second
            ):
                feat = extractor.extract_rgb_frame_features(rgb)
                if sum_rgb is None:
                    sum_rgb = feat.copy()
                else:
                    sum_rgb += feat
                rgb_features.append(_bytes_feature(quantize(feat)))

            if not rgb_features:
                print(f"Warning: Could not get features for {video_file}", file=sys.stderr)
                total_error += 1
                continue

            mean_rgb = sum_rgb / len(rgb_features)

            # Construir SequenceExample
            context_feats = {
                args.labels_feature_key: _int64_list_feature(
                    sorted(map(int, labels.split(';')))
                ),
                args.video_file_feature_key: _bytes_feature(
                    _make_bytes(list(map(ord, video_file)))
                ),
                'mean_' + args.image_feature_key:
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=mean_rgb.tolist())
                    )
            }

            feature_lists = {}
            if not args.skip_frame_level_features:
                feature_lists[args.image_feature_key] = tf.train.FeatureList(
                    feature=rgb_features
                )
                if args.insert_zero_audio_features:
                    zero_audio = _make_bytes([0] * 128)
                    audio_feats = [
                        _bytes_feature(zero_audio) for _ in rgb_features
                    ]
                    feature_lists['audio'] = tf.train.FeatureList(feature=audio_feats)
                    context_feats['mean_audio'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.0] * 128)
                    )

            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=context_feats),
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists)
            )

            writer.write(example.SerializeToString())
            total_written += 1
            print(f"✓ Encoded {video_file} ({total_written} written)")

    writer.close()
    print(f"Successfully encoded {total_written} out of {total_written + total_error} videos.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
