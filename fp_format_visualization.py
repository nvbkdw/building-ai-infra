"""
Floating Point Format Visualization
Creates a comparison graph showing exponent range vs precision for various FP formats,
including subnormal number representations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def calculate_fp_format_ranges(exp_bits: int, mantissa_bits: int) -> Tuple[List[int], List[int]]:
    """
    Calculate the exponent ranges and corresponding precision for a given FP format.
    
    Args:
        exp_bits: Number of exponent bits (E)
        mantissa_bits: Number of mantissa bits (M)
    
    Returns:
        Tuple of (exponents, precisions) lists
    """
    # Calculate bias
    bias = 2 ** (exp_bits - 1) - 1
    
    exponents = []
    precisions = []
    
    # Subnormal range: exponent = -(bias + mantissa_bits) to -bias
    # Precision decreases as we go to smaller exponents
    min_subnormal_exp = -(bias + mantissa_bits)
    max_subnormal_exp = -bias
    
    # Add subnormal range (precision decreases)
    for i in range(mantissa_bits + 1):
        exp = min_subnormal_exp + i
        precision = i  # Precision increases from 0 to mantissa_bits
        exponents.append(exp)
        precisions.append(precision)
    
    # Normal range: exponent = -(bias-1) to (2^exp_bits - 2 - bias)
    min_normal_exp = -(bias - 1)
    max_normal_exp = 2 ** exp_bits - 2 - bias
    
    # Add transition point (first normal number)
    exponents.append(min_normal_exp)
    precisions.append(mantissa_bits)
    
    # Add max normal exponent
    exponents.append(max_normal_exp)
    precisions.append(mantissa_bits)
    
    return exponents, precisions


def plot_fp_formats(formats: Dict[str, Tuple[int, int]], 
                   output_file: str = 'fp-format-exp-precision.png',
                   figsize: Tuple[int, int] = (14, 8),
                   dpi: int = 100):
    """
    Create a comparison plot of different floating-point formats.
    
    Args:
        formats: Dictionary mapping format names to (exp_bits, mantissa_bits) tuples
                Example: {'fp32': (8, 23), 'fp16': (5, 10)}
        output_file: Path to save the output image
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
    """
    plt.figure(figsize=figsize, facecolor='lightgray')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    
    # Define colors for different formats
    colors = {
        'fp32': 'black',
        'fp16': 'red',
        'bf16': 'darkred',
        'e5m2': 'olive',
        'e4m3': 'blue',
        'e2m1': 'cyan',
    }
    
    # Plot each format
    for format_name, (exp_bits, mantissa_bits) in formats.items():
        exponents, precisions = calculate_fp_format_ranges(exp_bits, mantissa_bits)
        
        # Create step plot
        color = colors.get(format_name, 'gray')
        label = f'{format_name} (e{exp_bits}m{mantissa_bits})'
        
        plt.step(exponents, precisions, where='post', 
                linewidth=2.5, color=color, label=label)
    
    # Customize plot
    plt.xlabel('Exponent', fontsize=14, fontweight='bold')
    plt.ylabel('Precision (bits)', fontsize=14, fontweight='bold')
    plt.title('Floating Point Format Comparison: Exponent Range vs Precision', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, color='white', linewidth=1)
    
    # Set axis limits with some padding
    all_exponents = []
    all_precisions = []
    for exp_bits, mantissa_bits in formats.values():
        exps, precs = calculate_fp_format_ranges(exp_bits, mantissa_bits)
        all_exponents.extend(exps)
        all_precisions.extend(precs)
    
    x_min, x_max = min(all_exponents), max(all_exponents)
    y_min, y_max = min(all_precisions), max(all_precisions)
    
    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 1, y_max + 1)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', 
               facecolor='lightgray', edgecolor='none')
    print(f"Figure saved to: {output_file}")
    
    # Close the plot to free memory
    plt.close()


def main():
    """
    Main function to create the FP format comparison visualization.
    """
    # Define common floating-point formats
    # Format: name -> (exponent_bits, mantissa_bits)
    formats = {
        'fp32': (8, 23),   # IEEE 754 single precision
        'fp16': (5, 10),   # IEEE 754 half precision
        'bf16': (8, 7),    # Brain floating point (Google)
        'e5m2': (5, 2),    # FP8 E5M2
        'e4m3': (4, 3),    # FP8 E4M3
        'e2m1': (2, 1),    # Minimal FP format
    }
    
    # Create the visualization
    plot_fp_formats(
        formats=formats,
        output_file='static/fp-format-exp-precision.png',
        figsize=(14, 8),
        dpi=100
    )
    
    # Print format details
    print("\nFloating Point Format Details:")
    print("=" * 70)
    for name, (e, m) in formats.items():
        bias = 2 ** (e - 1) - 1
        min_subnormal = -(bias + m)
        max_normal = 2 ** e - 2 - bias
        print(f"{name:8s} (e{e}m{m}): "
              f"Subnormal: 2^{min_subnormal} to 2^-{bias}, "
              f"Normal: 2^-{bias-1} to 2^{max_normal}, "
              f"Precision: {m} bits")


if __name__ == '__main__':
    main()

