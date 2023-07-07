# all_layers = [
#     layer.name
#     for layer in reversed(model.get_layer("efficientnetv2-b0").layers)
#     if len(layer.output_shape) == 4
#     and (
#         layer.__class__.__name__ == "Activation"
#         or isinstance(layer, tf.keras.layers.DepthwiseConv2D)
#         or isinstance(layer, tf.keras.layers.Conv2D)
#     )
# ]

# # remove layers with "se"
# all_layers = [layer for layer in all_layers if "se" not in layer]

# predictions = model.predict(test_images).argmax(axis=1)

# grad_cam_images = []
# for i, layer in enumerate(X):
#     fused = fuse_layers(
#         all_layers, model.get_layer("efficientnetv2-b0"), test_images[i], True
#     )
#     grad_cam_images.append(fused)


# Change font to Times New Roman
# plt.rcParams["font.family"] = "Times New Roman"
# plt.figure(figsize=(15, 15), dpi=300)
# for i, image in enumerate(grad_cam_images):
#     ax = plt.subplot(7, 5, i + 1)
#     if CLASS_NAMES[test_labels[i]] is not CLASS_NAMES[predictions[i]]:
#         misclassified = True
#     else:
#         misclassified = False

#     plt.imshow(image)
#     plt.title(
#         f"True: {CLASS_NAMES[test_labels[i]]}\nPredicted: {CLASS_NAMES[predictions[i]]}",
#         color="red" if misclassified else "black",
#     )

#     plt.axis("off")

# plt.tight_layout()
