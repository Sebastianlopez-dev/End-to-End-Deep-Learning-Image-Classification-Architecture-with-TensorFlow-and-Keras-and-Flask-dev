# MobileNetV2 Test Suite: Image Difficulty Levels

This folder contains 10 generated images representing different levels of difficulty for the CIFAR-10 model, specifically targeting weaknesses identified in the project's model analysis.

## Level 1: Easy (Clear Distinctions)
These images represent standard, well-lit, unobstructed examples of their respective classes.
- **`01_easy_automobile.png`**: A bright red sports car on an open road. Distinct shape, no confusing background.
- **`02_easy_dog.png`**: A golden retriever sitting happily on a studio white background. Very clear animal features.
- **`03_easy_airplane.png`**: A clear passenger airplane centrally framed against a solid blue sky.

## Level 2: Medium (Higher Complexity)
These images include busy backgrounds, motion blur, partial occlusion, or less-ideal lighting.
- **`04_medium_ship.png`**: A cargo ship on a dark, stormy ocean. Partial occlusion by waves and low lighting make it harder to extract features.
- **`05_medium_frog.png`**: A green frog camouflaged on a green leaf. The model has to distinguish the subject from a similarly colored background.
- **`06_medium_deer.png`**: A deer running rapidly through a forest, creating motion blur. Evaluates robustness to imperfect photos.

## Level 3: Hard (Close to Confusion)
These images directly challenge the most common misclassifications noted in `REPORT.md` (e.g., Cat vs. Dog, Automobile vs. Truck).
- **`07_hard_cat.png`**: A large, fluffy Maine Coon cat panting. Its dog-like features heavily test the model's ability to separate cats from dogs.
- **`08_hard_bird.png`**: A brightly colored orange bird perched among visually complex, similarly colored tropical flowers and green leaves. Tests the model's ability to distinguish the bird's shape from a highly textured and colorful background. https://www.biodiversidad.gob.mx/especies/aves-de-mexico/identificacion
- **`09_hard_truck.png`**: A sleek, modern pickup truck that resembles an SUV or automobile. Tests the model's boundary between similar vehicles.
- **`10_hard_horse.png`**: A horse standing in a stable but photographed from an extreme, distorting low angle. Tests the model's reliance on standard object proportions.
