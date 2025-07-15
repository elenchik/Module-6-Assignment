library(keras)
library(magrittr)
library(grid)

# loading and preprocessing the dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

cat("Train images:", dim(train_images), "\n")
cat("Test images:", dim(test_images), "\n")
train_images <- train_images / 255
test_images <- test_images / 255
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# building model (6 layers total)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# compiling model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy'
)

# training model
model %>% fit(train_images, train_labels, epochs = 5, validation_data = list(test_images, test_labels))

# evaluating model
score <- model %>% evaluate(test_images, test_labels)
cat("Test accuracy:", score$accuracy, "\n")

# predicting on two test images
predictions <- model %>% predict(test_images[1:2,,,drop=FALSE])
print(predictions)

