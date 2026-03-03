# Alternative Transfer Learning Models Reference

This document outlines the various pre-trained models considered for the CIFAR-10 image classification project, their key characteristics, and the reasons why they were NOT chosen over MobileNetV2.

---

## Models Presented in the Presentation (Slides 4 & 4.1)

### MobileNetV2 (Chosen Model - Slide 4)
*   **Characteristics:** Remarkably efficient (~3.4M parameters). Uses inverted residual blocks.
*   **Why it WAS chosen:** It strikes the perfect balance between high-accuracy feature extraction and computational efficiency. It trains quickly on standard hardware without requiring massive resources.

### VGG16 (Slide 4.1)
*   **Characteristics:** Massive size (~138M parameters). Classical deep CNN architecture with uniform simple convolutional layers.
*   **Why Not?** It is very slow to train on standard hardware and highly prone to overfitting on small 32x32 images like CIFAR-10.

### ResNet50 (Slide 4.1)
*   **Characteristics:** Powerful architecture (~25M parameters) that utilizes skip connections (residual blocks) to train very deep networks without vanishing gradients.
*   **Why Not?** Its complexity is often overkill for CIFAR-10, leading to unnecessarily long training times without proportional accuracy gains for such small images.

### InceptionV3 (Slide 4.1)
*   **Characteristics:** Uses "Inception modules" capable of looking at the same image with different receptive fields (filter sizes) simultaneously. (~24M parameters)
*   **Why Not?** Demands high computational resources and typically requires much larger input resolutions (default is 299x299) to be fully effective. Highly upscaling 32x32 to 299x299 is very resource-intensive.

---

## Other Notable Alternatives

### EfficientNet (e.g., EfficientNetB0)
*   **Characteristics:** Modern, highly optimized models that scale up the network's depth, width, and resolution evenly. (B0 has ~5.3M parameters)
*   **Why Not?** While a very valid alternative, MobileNetV2 is slightly older but exceptionally well-documented for beginners, and tends to train slightly faster on basic setups.

### DenseNet (e.g., DenseNet121)
*   **Characteristics:** Connects each layer to every other layer in a feed-forward fashion, creating strong feature reuse. (~8M parameters)
*   **Why Not?** They are heavier in memory usage and take much longer to train per epoch compared to MobileNetV2, despite often achieving higher accuracy.

### Xception
*   **Characteristics:** An extension of the Inception architecture that uses depthwise separable convolutions (similar to MobileNet). (~22M parameters)
*   **Why Not?** It is quite large and heavily optimized for the massive ImageNet dataset. Using it for 32x32 pixel images is often inefficient.

### NASNetMobile
*   **Characteristics:** An architecture discovered by an AI (Neural Architecture Search) designed specifically for mobile and resource-constrained devices.
*   **Why Not?** While an excellent alternative, MobileNet design (inverted residual blocks) is much simpler and more intuitive to explain in a presentation and learning environment.
