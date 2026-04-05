from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


INIT_LR = 1e-4
EPOCHS = 30
BS = 32


DIRECTORY = r"C:\Users\nekhe\OneDrive\Documents\Project 2\Dataset"
CATEGORIES = ["improper_mask", "N95", "Surgical", "without_mask"]


print("[INFO] loading images...")
data = []
labels = []

valid_ext=('.jpg','.jpeg','.png')

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for root, _, files in os.walk(path):
        for img in files:
            img_path = os.path.join(root, img)
            try:
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)
                labels.append(category)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image: {img_path} ({e})")


le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)


with open("label_encoder.pickle", "wb") as f:
    pickle.dump(le, f)


data = np.array(data, dtype="float32")
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)


baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CATEGORIES), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=max(1,len(trainX) // BS),
    validation_data=(testX, testY),
    validation_steps=max(1, len(testX) // BS),
    epochs=EPOCHS
)


print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=le.classes_))


print("[INFO] saving mask detector model...")
model.save("mask_detector.model.h5")


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
