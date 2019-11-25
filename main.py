import train
import tensorflow as tf
import mafs
import struct
import numpy as np
import glob
import scipy.io
import gzip
import functools
import random
from skimage import io, transform
import os
import datetime
import pathlib
import json

tf.random.set_seed(None)

def img_heatmap(compute_img_log_p_x, imgcre, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rows = imgcre.shape[0]-args.rand_box[0]
    cols = imgcre.shape[1] - args.rand_box[1]
    heatmap = np.zeros((np.int(rows/args.spacing), np.int(cols/args.spacing)))
    with tf.device(args.device):
        if type(args.vh) is not np.ndarray:
            im_breakup_array = np.zeros((np.int(cols / args.spacing), args.rand_box_size * args.rand_box_size * 3), dtype=np.float32)
            for i in range(np.int(rows/args.spacing)*args.spacing):
                if not i%args.spacing:
                    for j in range(np.int(cols/args.spacing)*args.spacing):
                        if not j%args.spacing:
                            im_breakup_array[np.int(j/args.spacing),:] = tf.reshape(tf.image.crop_to_bounding_box(imgcre, i, j, args.rand_box_size, args.rand_box_size)/128 - 1, [-1]).numpy()
                    heatmap[np.int(i/args.spacing), :] = compute_img_log_p_x(x_mb=im_breakup_array).numpy()
        else:
            for i in range(np.int(rows/args.spacing)*args.spacing):
                im_breakup_array = np.zeros((np.int(cols / args.spacing), args.vh.shape[1]), dtype=np.float32)
                if not i%args.spacing:
                    for j in range(np.int(cols/args.spacing)*args.spacing):
                        if not j%args.spacing:
                            im_breakup_array[np.int(j/args.spacing),:] = tf.squeeze(tf.matmul(tf.reshape(tf.image.crop_to_bounding_box(imgcre, i, j, args.rand_box_size, args.rand_box_size)/128 - 1, [1, -1]), args.vh)).numpy()
                    heatmap[np.int(i/args.spacing), :] = compute_img_log_p_x(x_mb=im_breakup_array).numpy()
    return heatmap

def img_preprocessing(imgcre, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop = tf.image.random_crop(imgcre, args.rand_box)
    rand_crop = rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5) ## dequantize
    rand_crop = tf.clip_by_value(rand_crop, 0, 255)
    if type(args.vh) is np.ndarray:
        return tf.squeeze(tf.matmul(tf.reshape(rand_crop/128 - 1, [1,-1]), args.vh))
    else:
        return tf.reshape(rand_crop/128 - 1, [-1])

def img_load(filename, args):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(args.img_size)), tf.int32)
    imgcre = tf.image.resize(imgc, size=imresize_)
    return imgcre

def load_dataset(args):

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x,args),axis=0) for x in trainval])
    cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    cont_data = np.vstack([np.expand_dims(img_load(x,args),axis=0) for x in cont_data])

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    if args.vh:
        cliplist = []
        for n in range(np.int(6*args.n_dims/train_data.shape[0])):
            cliplist.append(np.vstack([img_preprocessing(x, args) for x in train_data]))
        svdmat = np.vstack(cliplist)
        _, _, args.vh = scipy.linalg.svd(svdmat, full_matrices=False)
        # with tf.device(args.device):
        #     _, _, args.vh = tf.linalg.svd(svdmat, full_matrices=False)
        args.vh = args.vh[:,:args.svdnum]

    img_preprocessing_ = functools.partial(img_preprocessing, args=args)
    # img_preprocessing_ = dequantize

    dataset_train = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(train_data)#.float().to(args.device)
    dataset_test = dataset_test.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_cont = tf.data.Dataset.from_tensor_slices(cont_data)#.float().to(args.device)
    dataset_cont = dataset_cont.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    args.n_dims = img_preprocessing_(train_data[0]).shape[0]

    # args.n_dims = train.shape[1]
    return dataset_train, dataset_valid, dataset_test, dataset_cont

class parser_:
    pass

args = parser_()
args.early_stopping = 30
args.batch_dim = 500
args.check_every = 5
args.show_log = True
args.batch_size = 500
args.max_iterations = 20000
args.num_layers = 5
args.num_hidden = [100]
args.act = tf.nn.relu
args.vh = 0 #0 =no, 1=yes
args.prefetch_size = 1  # data pipeline prefetch buffer size
args.parallel = 16  # data pipeline parallel processes
args.img_size = 0.25;  ## resize img between 0 and 1
args.preserve_aspect_ratio = True;  ##when resizing
args.rand_box_init = 0.1  ##relative size of random box from image
args.manualSeed = None
args.device = '/gpu:0'  # '/gpu:0'
args.svdnum = 100
args.clip_norm = 0.1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data_loader_train, data_loader_valid, data_loader_test, data_loader_cont = load_dataset(args)

## build model
model = mafs.MaskedAutoregressiveFlow(args.n_dims, args.num_hidden, args.act, args.num_layers, batch_norm=False, args=args)

t = train.Trainer(model, args) ## only pass model but don't re-initialize for SCE

## MLE training
t.train(data_loader_train, data_loader_valid, data_loader_cont, early_stopping=args.early_stopping, check_every_N=args.check_every, show_log=args.show_log, max_iterations=args.max_iterations, saver_name='temp/tmp_model')
# t.train(sess, data.astype(np.float32), val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, saver_name='temp/tmp_model')

# # import matplotlib.pyplot as plt
# # import scipy.stats
# s = model.gen(sess, 5000)
# # out = model.eval(train_data, sess)
# out2 = model.eval(sess.run(val), sess)
# out3 = model.eval(sess.run(cont_data), sess)
# sout_ = model.eval(s,sess)
# # dist = scipy.stats.johnsonsu.fit(out)
# # out = (np.arcsinh((out - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# # out2 = (np.arcsinh((out2 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# # out3 = (np.arcsinh((out3 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# # sout = (np.arcsinh((sout_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])

args.spacing = 8

args.path = os.path.join(r'D:\pycharm_projects\GQC_images_tensorboard', 'MAF_layers{}_h{}_vh{}_resize{}_boxsize{}_{}'.format(
    args.num_layers, args.num_hidden, np.any(args.vh>0), args.img_size, args.rand_box_init,
    str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

print('Creating directory experiment..')
pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
with open(os.path.join(args.path, 'args.json'), 'w') as f:
    json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

print('Loading dataset..')
trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in trainval])
cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
cont_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in cont_data])

## heatmaps:  change args.spacing closer to 1 to get more spatial resolution on the heatmap
heat_map_train = []
heat_map_train.extend(img_heatmap(model, img, args) for img in train_data)
heatmap_t = np.array(heat_map_train)
heat_map_cont = []
heat_map_cont.extend(img_heatmap(model, img, args) for img in cont_data)
heatmap_c = np.array(heat_map_cont)

## summaries
mean_t = np.mean(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
std_t = np.std(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
min_t = np.min(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
max_t = np.max(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)

mean_c = np.mean(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
std_c = np.std(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
min_c = np.min(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
max_c = np.max(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)

totals_t = np.stack([mean_t, std_t, min_t, max_t]).T
totals_c = np.stack([mean_c, std_c, min_c, max_c]).T

np.savetxt(args.path + '/totals_t' + '_' + str(args.spacing) + '_' + str(heatmap_t.reshape((heatmap_t.shape[0], -1)).shape[1]) + '.csv', totals_t, delimiter=',')
np.savetxt(args.path + '/totals_c' + '_' + str(args.spacing) + '_' + str(heatmap_c.reshape((heatmap_c.shape[0], -1)).shape[1]) + '.csv', totals_c, delimiter=',')

## heatmap arrays
np.save(args.path + '/totals_t' + '_' + str(args.spacing) + '_' + str(heatmap_t.reshape((heatmap_t.shape[0], -1)).shape[1]) + '.npy', heatmap_t)
np.save(args.path + '/totals_c' + '_' + str(args.spacing) + '_' + str(heatmap_c.reshape((heatmap_c.shape[0], -1)).shape[1]) + '.npy', heatmap_c)


# ############# debug
ins = [x for x in data_loader_train]
model.input = ins[0]
with tf.GradientTape() as tape:
    loss = model.trn_loss()
grads = tape.gradient(loss, model.parms)
grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
t.optimizer.apply_gradients(zip(grads, model.parms))