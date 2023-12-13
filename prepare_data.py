import os
import numpy as np
import argparse
from sklearn.cluster import MiniBatchKMeans
import pickle
import time
from datetime import timedelta, datetime
def save_pickle(path, obj):
    """
    simple method to save a picklable object
    :param path: path to save
    :param obj: a picklable object
    :return: None
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class Timer(object):

    def __init__(self):
        self.start_t = time.time()
        self.last_t = self.start_t

    def time(self, lap=False):
        end_t = time.time()
        if lap:
            out = timedelta(seconds=int(end_t - self.last_t))  # count from last stop point
        else:
            out = timedelta(seconds=int(end_t - self.start_t))  # count from beginning
        self.last_t = end_t
        return out


def load_data(data_dir, class_names):
    """
    Load all sketch data into two arrays (pen_state=0 and pen_state=1)
    :param data_dir: Directory containing the .npz files for each class
    :param class_names: List of class names
    :return: Two arrays, each containing data for one pen state
    """
    out_p0, out_p1 = [], []
    for class_name in class_names:
        file_path = os.path.join(data_dir, f'{class_name}.npz')
        if os.path.exists(file_path):
            data = np.load(file_path, encoding='latin1', allow_pickle=True)
            total_samples = data['train']
            total_samples = len(data['train'])
            quarter_samples = total_samples // 4  # Integer division to get a quarter of the total

            # Take the first quarter of the samples
            samples = data['train'][:quarter_samples]
            for sketch in samples:
                sketch = normalize_sketch(sketch)
                pen_lift_ids = np.where(sketch[:, 2] == 1)[0] + 1
                pen_lift_ids = pen_lift_ids[:-1]
                pen_hold_ids = set(range(len(sketch))) - set(pen_lift_ids)

                out_p0.append(sketch[list(pen_hold_ids), :2])
                out_p1.append(sketch[list(pen_lift_ids), :2])

    return np.concatenate(out_p0), np.concatenate(out_p1)

def normalize_sketch(sketch):
    # Normalize sketch data
    sketch = np.minimum(sketch, 1000)
    sketch = np.maximum(sketch, -1000)
    min_x, max_x, min_y, max_y = get_bounds(sketch)
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    sketch = sketch.astype(np.float32)
    sketch[:, :2] /= max_dim
    return sketch

def get_bounds(sketch):
    min_x = np.min(sketch[:, 0])
    max_x = np.max(sketch[:, 0])
    min_y = np.min(sketch[:, 1])
    max_y = np.max(sketch[:, 1])
    return min_x, max_x, min_y, max_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a dictionary of sketch tokens')
    parser.add_argument('--data-dir', type=str, default='/content/quickdraw', help='Directory containing .npz files')
    parser.add_argument('--class-list', type=str, help='List of class names')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--num-samples', type=int, default=5000000, help='Number of samples')
    parser.add_argument('--p1-ratio', type=float, default=0.2, help='Ratio of pen_state=1 samples')
    parser.add_argument('--target-file', type=str, default='token_dict.pkl', help='Output token dictionary file')

    args = parser.parse_args()
    timer = Timer()
    class_names = []
    with open(args.class_list) as clf:
        class_names = clf.read().splitlines()

    print("Loading data ...")
    data_p0, data_p1 = load_data(args.data_dir, class_names)
    N_P0, N_P1 = data_p0.shape[0], data_p1.shape[0]
    print("p1/p0 natural ratio: %f" % (N_P1 / N_P0))
    
    n_p1 = int(args.p1_ratio * args.num_samples)
    n_p0 = args.num_samples - n_p1
    if N_P0 > n_p0:
        print("Sample %d out of %d points with pen_state 0" % (n_p0, N_P0))
        ids_p0 = np.random.choice(N_P0, n_p0, replace=False)
        data_p0 = data_p0[ids_p0]
    if N_P1 > n_p1:
        print("Sample %d out of %d points with pen_state 1" % (n_p1, N_P1))
        ids_p1 = np.random.choice(N_P1, n_p1, replace=False)
        data_p1 = data_p1[ids_p1]

    data = np.r_[data_p0, data_p1]

    N = data.shape[0]
    print("Building dictionary ...")
    cluster = MiniBatchKMeans(n_clusters=args.vocab_size, max_iter=200, compute_labels=False,
                                   verbose=1, n_init=5).fit(data)
    print("Dictionary built: {}".format(timer.time(True)))
    save_pickle(args.target_file, cluster)
    print("Total time: {}".format(timer.time(False)))