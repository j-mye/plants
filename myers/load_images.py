from PIL import Image
import numpy as np
import os
import glob
import tensorflow as tf

# Set up paths and parameters
IMG_SIZE = 224  # MobileNet works well with 224x224 images
BATCH_SIZE = 32
train_dir = "../datasets/dataset"
test_dir = "../datasets/dataset-test"

# Get the number of classes from the training directory
# Each subdirectory represents a different plant class
class_names = sorted([d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)

print(f"Found {num_classes} plant classes")
print(f"Sample classes: {class_names[:5]}...")

def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Load an image and preprocess it maintaining aspect ratio.
    This prevents stretching/distortion of images.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the longest side
    
    Returns:
        Processed image array (target_size x target_size) with aspect ratio preserved
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get original dimensions
    orig_width, orig_height = img.size
    
    # Calculate new dimensions maintaining aspect ratio
    # Resize so longest side becomes target_size
    if orig_width > orig_height:
        new_width = target_size
        new_height = int(target_size * orig_height / orig_width)
    else:
        new_height = target_size
        new_width = int(target_size * orig_width / orig_height)
    
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = getattr(Image, 'LANCZOS', Image.ANTIALIAS)

    img_resized = img.resize((new_width, new_height), resample_filter)
    
    # Create white background canvas
    img_padded = Image.new('RGB', (target_size, target_size), color=(255, 255, 255))
    
    # Center the resized image on the canvas
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    img_padded.paste(img_resized, (offset_x, offset_y))
    
    # Convert to numpy array and normalize to [0, 1]
    return np.array(img_padded, dtype=np.float32) / 255.0

# Create a custom generator that properly handles aspect ratio
def create_image_generator(directory, batch_size, shuffle=True, augment=False):
    """
    Create a generator that loads images maintaining aspect ratio.
    Returns a generator and the sample count.
    """
    # Get all image files (scan once, outside the generator)
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Find all image files — avoid double-globbing by collecting in one pass
        class_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            class_images.extend(glob.glob(os.path.join(class_dir, ext)))
        
        # Remove duplicates (in case .jpg and .JPG both match same file)
        class_images = list(set(class_images))
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    total_samples = len(image_paths)
    
    # Shuffle if needed
    if shuffle:
        indices = np.random.permutation(total_samples)
        image_paths = image_paths[indices]
        labels = labels[indices]
    
    # Create one-hot encoded labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
    
    # Generator function: infinite loop for Keras compatibility
    def generator():
        while True:  # Infinite loop so Keras can request as many batches as needed
            # Shuffle again each epoch if requested
            if shuffle:
                epoch_indices = np.random.permutation(total_samples)
                epoch_paths = image_paths[epoch_indices]
                epoch_labels = labels_onehot[epoch_indices]
            else:
                epoch_paths = image_paths
                epoch_labels = labels_onehot
            
            # Yield one full epoch per cycle
            for i in range(0, total_samples, batch_size):
                batch_paths = epoch_paths[i:i+batch_size]
                batch_labels = epoch_labels[i:i+batch_size]
                
                # Load and preprocess images
                batch_images = []
                for path in batch_paths:
                    img = load_and_preprocess_image(path)
                    
                    # Apply augmentation if training
                    if augment:
                        # Horizontal flip
                        if np.random.random() > 0.5:
                            img = np.fliplr(img)
                        # Random rotation (simple version)
                        if np.random.random() > 0.5:
                            angle = np.random.uniform(-20, 20)
                            # Convert to PIL for rotation, then back
                            img_pil = Image.fromarray((img * 255).astype(np.uint8))
                            # Use safe rotate with fillcolor fallback
                            try:
                                img_pil = img_pil.rotate(angle, fillcolor=(255, 255, 255))
                            except TypeError:
                                # Older Pillow doesn't support fillcolor
                                img_pil = img_pil.rotate(angle)
                            img = np.array(img_pil, dtype=np.float32) / 255.0
                    
                    batch_images.append(img)
                
                yield np.array(batch_images), batch_labels
    
    return generator(), total_samples

# Create generators
train_gen, train_samples = create_image_generator(
    train_dir, BATCH_SIZE, shuffle=True, augment=True
)
test_gen, test_samples = create_image_generator(
    test_dir, BATCH_SIZE, shuffle=False, augment=False
)

print(f"\nTraining samples: {train_samples}")
print(f"Test samples: {test_samples}")
print(f"Number of classes: {num_classes}")
print(f"\n✓ Images are resized maintaining aspect ratio (no stretching!)")
print(f"  Longest side → {IMG_SIZE}px, padded to {IMG_SIZE}x{IMG_SIZE} square")
