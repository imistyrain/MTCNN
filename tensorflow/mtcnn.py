import tensorflow as tf
import cv2

class MTCNN:
    def __init__(self, model_path, min_size=40, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds

        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

    def detect(self, img):
        feeds = {
            self.graph.get_operation_by_name('input').outputs[0]: img,
            self.graph.get_operation_by_name('min_size').outputs[0]: self.min_size,
            self.graph.get_operation_by_name('thresholds').outputs[0]: self.thresholds,
            self.graph.get_operation_by_name('factor').outputs[0]: self.factor
        }
        fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                  self.graph.get_operation_by_name('landmarks').outputs[0],
                  self.graph.get_operation_by_name('box').outputs[0]]
        prob, landmarks, box = self.sess.run(fetches, feeds)
        return box, prob, landmarks

    def detectAndDraw(self,img):
        bbox, scores, landmarks = self.detect(img)
        print('total box:', len(bbox))
        for box, pts in zip(bbox, landmarks):
            box = box.astype('int32')
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            pts = pts.astype('int32')
            for i in range(5):
                img = cv2.circle(img, (pts[i+5], pts[i]), 1, (0, 255, 0), 2)
        return img