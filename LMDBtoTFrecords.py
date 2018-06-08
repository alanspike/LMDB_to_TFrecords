import numpy as np
import imp
import lmdb
import tensorflow as tf

import argparse

def _bytes_feature(value):
	"""
	For Image Files
	Args:
		value: input image
	"""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
	"""
	For Label:
	Args:
		value: input label
	"""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def LMDBtoTFrecord(lmdb_path, output_path):
	"""
	Read a Caffe LMDB file where each value contains a ``caffe.Datum`` protobuf.
	Produces datapoints of the format: [HWC image, label].
	Save the data to tfrecords file.
	"""

	lmdb_env = lmdb.open(lmdb_path)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()

	cpb = imp.load_source('caffepb', 'caffe_pb2.py')


	with tf.python_io.TFRecordWriter(output_path) as record_writer::
		for key, value in lmdb_cursor:
			datum = cpb.Datum()
			datum.ParseFromString(value)
			label = datum.label

			img = np.array(datum.float_data).astype(np.float32).reshape(datum.channels, datum.height, datum.width)
			img = np.transpose(img, (1, 2, 0)) # original (dim, col, row)

			img = img.tostring()

			example = tf.train.Example(features=tf.train.Features(
				feature={
					'image': _bytes_feature(tf.compat.as_bytes(img)),
					'label': _int64_feature(label)
			}))

			record_writer.write(example.SerializeToString())


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--lmdb_path',
						help='the input lmdb folder',
						default=None,
						type=str)
    
    parser.add_argument('--output_path',
						help='the output *.tfrecords',
						default=None,
						type=str)

    args = parser.parse_args()


    LMDBtoTFrecord(args.lmdb_path, args.output_path)

	

