import tensorflow as tf


# 从tensorflow官方下载的压缩包解压后，里面有训练模型文件，权重文件，以及冻结的pb文件(protobuf)，还有针对移动设备的tflite文件
# 但是我们需要的只有三个文件：ckpt.data - 权重文件，ckpt.meta - 计算图文件，ckpt.index - 模型结构的变量和参数之间的索引对应关系
# 将需要的三个文件放到pretrain_model文件夹中
def rename_var(ckpt_path, new_ckpt_path):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        for var_name, shape in var_list:
            # print(var_name)
            if var_name in except_list:
                continue
            # 因为在模型的权重保存过程中，也把优化器的参数都保留了，这些参数我们也不需要
            if "RMSProp" in var_name or "Exponential" in var_name:
                continue
            var = tf.train.load_variable(ckpt_path, var_name)
            # 将权重名称改为我们自己定义的模型中的名称
            new_var_name = var_name.replace('MobilenetV2/', "")
            new_var_name = new_var_name.replace("/expand/weights", "/expand/Conv2d/weights")
            new_var_name = new_var_name.replace("Conv/weights", "Conv/Conv2d/kernel")
            new_var_name = new_var_name.replace("Conv_1/weights", "Conv_1/Conv2d/kernel")
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")

            first_word = new_var_name.split('/')[0]
            if "expanded_conv" in first_word:
                last_word = first_word.split('expanded_conv')[-1]
                if len(last_word) > 0:
                    new_word = "inverted_residual" + last_word + "/expanded_conv/"
                else:
                    new_word = "inverted_residual/expanded_conv/"
                new_var_name = new_word + new_var_name.split('/', maxsplit=1)[-1]
            print(new_var_name)
            re_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(re_var)

        # 重新定义全连接层
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([1280, 5]), name="Logits/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([5]), name="Logits/bias")
        new_var_list.append(re_var)
        # 在resnet的read_ckpt.py脚本中没有下面这一行，但是也执行成功了，不知道为什么
        tf.keras.initializers.he_uniform()
        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


# 预训练的mobilenet网络最后是用1*1卷积层来做预测的，我们自己定义的网络是用全连接层来预测的
except_list = ['global_step', 'MobilenetV2/Logits/Conv2d_1c_1x1/biases', 'MobilenetV2/Logits/Conv2d_1c_1x1/weights']
ckpt_path = './pretrain_model/mobilenet_v2_1.0_224.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
new_var_list = []
rename_var(ckpt_path, new_ckpt_path)
