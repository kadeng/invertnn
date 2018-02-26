from unittest import TestCase
from invertnn.tf.graph_manipulation import extract_intermediate_graph
from invertnn.general.gpu_usage import grab_gpus
grab_gpus(0)
import tensorflow as tf
import numpy


class TestGraphManipulation(TestCase):


    def test_invertible1(self):
        rng = numpy.random

        with tf.Graph().as_default():
            g = tf.get_default_graph()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)):
                session = tf.get_default_session()

                # tf Graph Input
                X1 = tf.placeholder("float", name='X1')
                X2 = tf.placeholder("float", name='X1')
                X = tf.identity(X1 + X2 * X2, name='X')

                # Set model weights
                W = tf.Variable(numpy.ones((1,), dtype=numpy.float32), name="weight")
                b = tf.Variable(numpy.ones((1,), dtype=numpy.float32), name="bias")

                # Construct a linear model
                Y = tf.identity(tf.add(tf.multiply(X, W), b), name='Y')

                init = tf.global_variables_initializer()
                session.run(init)

                frozen_graph = tf.graph_util.convert_variables_to_constants(
                    sess=session, # The session is used to retrieve the weights
                    input_graph_def=g.as_graph_def(), # The graph_def is used to retrieve the nodes
                    output_node_names=['Y'])
                igraph = extract_intermediate_graph(g, frozen_graph, ['X'], ['Y'])
                igr = str(igraph)#
                self.assertTrue(igr.strip()=="""
node {
  name: "X"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "weight"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "weight/read"
  op: "Identity"
  input: "weight"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@weight"
      }
    }
  }
}
node {
  name: "bias"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "bias/read"
  op: "Identity"
  input: "bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@bias"
      }
    }
  }
}
node {
  name: "Mul_1"
  op: "Mul"
  input: "X"
  input: "weight/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Add_1"
  op: "Add"
  input: "Mul_1"
  input: "bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Y"
  op: "Identity"
  input: "Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
library {
}
versions {
}
""".strip())


