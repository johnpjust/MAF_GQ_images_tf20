# MAF_GQ_images_tf20
tf2.0 Implementation of Masked Autoregressive Flows for Density Estimation

There's probably some room for optimization here, such as using tf.function.  Runs a bit slower than graph mode and the LL is less.  In graph mode the LL obtained was > BNAF slightly, now it is slightly less than it.  From previous work though higher LL doesn't ensure there is a tangible/noticeable difference for most applications.
