
#Alguns parâmetros de configuração do modelo de regressão linear

Number of training examples: m_train = " + str(m_train))\
Number of testing examples: m_test = " + str(m_test))\
Height/Width of each image: num_px = " + str(num_px))
Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
train_set_x shape: " + str(train_set_x_orig.shape))
train_set_y shape: " + str(train_set_y.shape))
test_set_x shape: " + str(test_set_x_orig.shape))
test_set_y shape: " + str(test_set_y.shape))


train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
train_set_y shape: " + str(train_set_y.shape))
test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
test_set_y shape: " + str(test_set_y.shape))
sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
