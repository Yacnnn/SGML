                    XX = tf.einsum('ij,ij->i', output[i], output[i])[:, np.newaxis]
                    YY = tf.einsum('ij,ij->i', output[j], output[j])[np.newaxis, :]
                    distances = tf.tensordot(output[i], tf.transpose(output[j]), [[1],[0]])
                    distances *= -2
                    distances += XX
                    distances += YY
                    distances =  tf.nn.relu(distances)