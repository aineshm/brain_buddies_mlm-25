#!/usr/bin/env python3
"""
Comprehensive Gaussian Blur Explanation and Demonstration
Shows how Gaussian blurring works mathematically and visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, data
from scipy import ndimage
import tifffile
import sys
import os

# Add parent directory to path for imports
sys.path.append('..')
try:
    from xml_shape_parser import XMLShapeParser
except ImportError:
    print("Note: xml_shape_parser not available, using sample data")

class GaussianBlurDemo:
    """Comprehensive demonstration of Gaussian blur effects."""
    
    def __init__(self):
        """Initialize the demo."""
        self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Configure matplotlib for better visualization."""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
    
    def create_gaussian_kernel_2d(self, size=15, sigma=2.0):
        """
        Create a 2D Gaussian kernel manually to show the math.
        
        Args:
            size: Kernel size (odd number)
            sigma: Standard deviation
        
        Returns:
            2D numpy array representing the Gaussian kernel
        """
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
        
        # Create coordinate arrays
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        
        # Apply Gaussian formula: G(x,y) = (1/(2πσ²)) * e^(-(x² + y²)/(2σ²))
        kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize so sum equals 1 (preserve image brightness)
        kernel = kernel / np.sum(kernel)
        
        return kernel, x, y
    
    def demonstrate_kernel_creation(self):
        """Show how Gaussian kernels are created with different sigma values."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Gaussian Kernel Creation: Effect of Sigma (σ) Values', fontsize=16, fontweight='bold')
        
        sigmas = [0.5, 1.0, 2.0]
        kernel_size = 15
        
        for i, sigma in enumerate(sigmas):
            # Create kernel
            kernel, x, y = self.create_gaussian_kernel_2d(kernel_size, sigma)
            
            # 3D surface plot
            ax1 = axes[0, i]
            im1 = ax1.imshow(kernel, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'σ = {sigma}\nKernel Size: {kernel_size}x{kernel_size}')
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Cross-section through center
            ax2 = axes[1, i]
            center = kernel_size // 2
            cross_section = kernel[center, :]
            x_coords = np.arange(kernel_size) - center
            
            ax2.plot(x_coords, cross_section, 'b-', linewidth=2, marker='o')
            ax2.set_title(f'Cross-section (σ = {sigma})')
            ax2.set_xlabel('Distance from center')
            ax2.set_ylabel('Kernel value')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gaussian_kernels_explanation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return sigmas
    
    def load_cell_image(self):
        """Load a cell image for demonstration."""
        try:
            # Try to load the training TIF
            tif_path = '../training.tif'
            if os.path.exists(tif_path):
                print(f"Loading cell image: {tif_path}")
                try:
                    tif_data = tifffile.imread(tif_path)
                    if len(tif_data.shape) == 3:
                        # Use frame 0 as our test image
                        image = tif_data[0]
                    else:
                        image = tif_data
                    print(f"Loaded image shape: {image.shape}")
                    return image
                except Exception as e:
                    print(f"TIF loading error: {e}")
            
            # Fallback to sample data
            print("Using sample data from scikit-image")
            image = data.camera()
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create synthetic cell-like image
            return self.create_synthetic_cell_image()
    
    def create_synthetic_cell_image(self):
        """Create a synthetic cell-like image for demonstration."""
        print("Creating synthetic cell-like image")
        
        # Create base image
        size = 200
        image = np.zeros((size, size))
        
        # Add some cell-like structures
        for i in range(5):
            # Random cell center
            cx, cy = np.random.randint(30, size-30, 2)
            # Random cell size
            radius = np.random.randint(15, 25)
            
            # Create circular cell
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            image[mask] = np.random.uniform(0.6, 1.0)
        
        # Add noise
        noise = np.random.normal(0, 0.1, (size, size))
        image = image + noise
        
        # Normalize
        image = np.clip(image, 0, 1)
        
        return image
    
    def demonstrate_blur_effects(self, image):
        """Show the effect of different blur strengths."""
        print("Demonstrating blur effects...")
        
        # Different sigma values to test
        sigmas = [0, 0.5, 1.0, 2.0, 4.0, 8.0]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gaussian Blur Effects: Different Sigma Values', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, sigma in enumerate(sigmas):
            if sigma == 0:
                # Original image
                blurred = image
                title = 'Original (σ = 0)'
            else:
                # Apply Gaussian blur
                blurred = filters.gaussian(image, sigma=sigma)
                title = f'σ = {sigma}'
            
            # Display
            axes_flat[i].imshow(blurred, cmap='gray', vmin=0, vmax=1)
            axes_flat[i].set_title(title)
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('gaussian_blur_effects.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return sigmas
    
    def demonstrate_noise_reduction(self, image):
        """Show how Gaussian blur reduces different types of noise."""
        print("Demonstrating noise reduction...")
        
        # Add different types of noise
        noisy_images = {}
        
        # Salt and pepper noise
        salt_pepper = image.copy()
        noise_mask = np.random.random(image.shape) < 0.05
        salt_pepper[noise_mask] = np.random.choice([0, 1], size=np.sum(noise_mask))
        noisy_images['Salt & Pepper'] = salt_pepper
        
        # Gaussian noise
        gaussian_noise = image + np.random.normal(0, 0.15, image.shape)
        gaussian_noise = np.clip(gaussian_noise, 0, 1)
        noisy_images['Gaussian Noise'] = gaussian_noise
        
        # Speckle noise
        speckle = image + image * np.random.normal(0, 0.1, image.shape)
        speckle = np.clip(speckle, 0, 1)
        noisy_images['Speckle Noise'] = speckle
        
        # Test different blur strengths
        sigmas = [1.0, 2.0, 3.0]
        
        fig, axes = plt.subplots(len(noisy_images), len(sigmas) + 1, figsize=(20, 12))
        fig.suptitle('Gaussian Blur for Noise Reduction', fontsize=16, fontweight='bold')
        
        for row, (noise_type, noisy_img) in enumerate(noisy_images.items()):
            # Show original noisy image
            axes[row, 0].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_title(f'{noise_type}\n(Original)')
            axes[row, 0].axis('off')
            
            # Show blur effects
            for col, sigma in enumerate(sigmas):
                blurred = filters.gaussian(noisy_img, sigma=sigma)
                axes[row, col + 1].imshow(blurred, cmap='gray', vmin=0, vmax=1)
                axes[row, col + 1].set_title(f'σ = {sigma}')
                axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('gaussian_noise_reduction.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def demonstrate_edge_preservation(self, image):
        """Show how blur affects edge preservation."""
        print("Demonstrating edge preservation...")
        
        # Create an image with sharp edges for demonstration
        edge_image = np.zeros((100, 100))
        edge_image[25:75, 25:75] = 1.0  # White square
        
        sigmas = [0, 0.5, 1.0, 2.0, 4.0]
        
        fig, axes = plt.subplots(2, len(sigmas), figsize=(20, 8))
        fig.suptitle('Edge Preservation vs Blur Strength', fontsize=16, fontweight='bold')
        
        for i, sigma in enumerate(sigmas):
            if sigma == 0:
                blurred = edge_image
                title = 'Original'
            else:
                blurred = filters.gaussian(edge_image, sigma=sigma)
                title = f'σ = {sigma}'
            
            # Show blurred image
            axes[0, i].imshow(blurred, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(title)
            axes[0, i].axis('off')
            
            # Show cross-section through center
            center_line = blurred[50, :]
            axes[1, i].plot(center_line, 'b-', linewidth=2)
            axes[1, i].set_title('Cross-section')
            axes[1, i].set_ylim(0, 1)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gaussian_edge_preservation.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_summary(self):
        """Create a comprehensive summary figure."""
        print("Creating comprehensive summary...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create text summary
        summary_text = """
GAUSSIAN BLURRING: COMPLETE EXPLANATION

🔍 MATHEMATICAL FOUNDATION:
   • Gaussian Function: G(x,y) = (1/(2πσ²)) × e^(-(x² + y²)/(2σ²))
   • σ (sigma): Controls blur strength
   • Larger σ = More blur
   • Kernel normalized so sum = 1 (preserves brightness)

🎯 KEY PROPERTIES:
   • Smoothing: Reduces high-frequency noise
   • Isotropic: Blurs equally in all directions
   • Separable: Can be applied as two 1D operations (faster)
   • Linear: Preserves image linearity

⚡ APPLICATIONS IN CELL SEGMENTATION:
   • Noise Reduction: Removes background granularity
   • Preprocessing: Prepares images for edge detection
   • Multi-scale Analysis: Different σ values capture different features
   • U-Net Input: Provides smoother input for neural networks

🔧 PARAMETER SELECTION:
   • σ = 0.5-1.0: Light smoothing, preserves fine details
   • σ = 1.0-2.0: Moderate smoothing, good for noise reduction
   • σ = 2.0-4.0: Strong smoothing, removes fine structures
   • σ > 4.0: Very strong smoothing, may lose important features

✅ ADVANTAGES:
   • Mathematically well-defined
   • Preserves image structure better than other filters
   • Fast computation (especially separable implementation)
   • No ringing artifacts

⚠️ CONSIDERATIONS:
   • Removes fine details along with noise
   • May blur important cellular boundaries
   • Parameter selection critical for good results
   • Trade-off between noise reduction and feature preservation
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.axis('off')
        plt.title('Gaussian Blurring: Mathematical and Practical Guide', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('gaussian_blur_comprehensive_guide.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_complete_demonstration(self):
        """Run the complete Gaussian blur demonstration."""
        print("=" * 80)
        print("GAUSSIAN BLUR: COMPLETE DEMONSTRATION")
        print("=" * 80)
        
        # 1. Kernel creation
        print("\n1. Demonstrating Gaussian kernel creation...")
        self.demonstrate_kernel_creation()
        
        # 2. Load test image
        print("\n2. Loading test image...")
        image = self.load_cell_image()
        
        # 3. Blur effects
        print("\n3. Demonstrating blur effects...")
        self.demonstrate_blur_effects(image)
        
        # 4. Noise reduction
        print("\n4. Demonstrating noise reduction...")
        self.demonstrate_noise_reduction(image)
        
        # 5. Edge preservation
        print("\n5. Demonstrating edge preservation...")
        self.demonstrate_edge_preservation(image)
        
        # 6. Summary
        print("\n6. Creating comprehensive summary...")
        self.create_comprehensive_summary()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("Generated files:")
        print("  • gaussian_kernels_explanation.png")
        print("  • gaussian_blur_effects.png") 
        print("  • gaussian_noise_reduction.png")
        print("  • gaussian_edge_preservation.png")
        print("  • gaussian_blur_comprehensive_guide.png")
        print("=" * 80)

def main():
    """Main function to run the demonstration."""
    demo = GaussianBlurDemo()
    demo.run_complete_demonstration()

if __name__ == "__main__":
    main()