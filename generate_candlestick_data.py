# Krutik Bajariya & Rahil Jhaveri
# CS 5330 - PRCV - Final Project
# generate_candlestick_data.py
# This script generates a dataset of stock chart patterns using historical stock data.
# It downloads stock data from Yahoo Finance, detects various chart patterns, and saves the patterns as images with YOLO format annotations.
# The detected patterns include:
# - Head and Shoulders Top
# - Head and Shoulders Bottom
# - M Head (Double Top)
# - W Bottom (Double Bottom)
# - Triangle
# - StockLine
# The script uses the yfinance library to download stock data, and matplotlib to generate candlestick charts.

import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from scipy.signal import find_peaks

def generate_pattern_dataset(tickers, period="5y", interval="1d", num_samples_per_pattern=100, output_dir="pattern_dataset"):
    """
    Generate a dataset of stock chart patterns

    Args:
        tickers (list): List of stock ticker symbols
        period (str): Time period to download data for (e.g., '5y')
        interval (str): Data interval (e.g., '1d')
        num_samples_per_pattern (int): Number of samples to generate per pattern
        output_dir (str): Directory to save the dataset

    Returns:
        list: Paths to the generated chart images with pattern annotations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define patterns we want to detect
    pattern_types = [
        'Head and shoulders bottom',
        'Head and shoulders top',
        'M_Head',
        'StockLine',
        'Triangle',
        'W_Bottom'
    ]
    
    # Create pattern-specific directories
    for pattern_name in pattern_types:
        os.makedirs(os.path.join(output_dir, pattern_name.replace(' ', '_')), exist_ok=True)
    
    generated_charts = []
    
    # Number of patterns detected per type
    pattern_counts = {pattern_name: 0 for pattern_name in pattern_types}
    
    # Loop through tickers to get data
    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            # Get data
            data = yf.download(ticker, period=period, interval=interval)
            data.reset_index(drop=True, inplace=True)
            data.columns = ['Close', 'High', 'Low', 'Open', 'Volumne']
            
            if len(data) < 50:  # Skip if not enough data
                continue
                
            print(f"Downloaded {len(data)} days of data for {ticker}")
            
            # Preprocess data
            data = data.fillna(method='ffill')
            
            # Find peaks and troughs
            window_size = 5
            prominence = 0.5
            
            prices = data['Close'].values
            peaks, _ = find_peaks(prices, distance=window_size, prominence=prominence * np.mean(prices) / 10)
            troughs, _ = find_peaks(-prices, distance=window_size, prominence=prominence * np.mean(prices) / 10)
            
            # Detect patterns
            patterns = {}
            
            # Head and shoulders top
            hs_top = detect_head_and_shoulders_top(data, peaks, troughs)
            patterns['Head and shoulders top'] = hs_top
            pattern_counts['Head and shoulders top'] += len(hs_top)
            
            # Head and shoulders bottom
            hs_bottom = detect_head_and_shoulders_bottom(data, peaks, troughs)
            patterns['Head and shoulders bottom'] = hs_bottom
            pattern_counts['Head and shoulders bottom'] += len(hs_bottom)
            
            # M-Head (Double Top)
            m_head = detect_m_head(data, peaks, troughs)
            patterns['M_Head'] = m_head
            pattern_counts['M_Head'] += len(m_head)
            
            # W-Bottom (Double Bottom)
            w_bottom = detect_w_bottom(data, peaks, troughs)
            patterns['W_Bottom'] = w_bottom
            pattern_counts['W_Bottom'] += len(w_bottom)
            
            # Triangle
            triangle = detect_triangle(data, peaks, troughs)
            patterns['Triangle'] = triangle
            pattern_counts['Triangle'] += len(triangle)
            
            # StockLine
            stock_line = detect_stock_line(data)
            patterns['StockLine'] = stock_line
            pattern_counts['StockLine'] += len(stock_line)
            
            # Generate images for each pattern
            for pattern_name, pattern_list in patterns.items():
                if not pattern_list:
                    continue
                    
                print(f"Found {len(pattern_list)} {pattern_name} patterns in {ticker}")
                
                # If patterns are found, generate images
                # Select random samples if there are more than needed
                samples_per_ticker = min(
                    num_samples_per_pattern // len(tickers), 
                    len(pattern_list)
                )
                samples_per_ticker = max(1, samples_per_ticker)  # Ensure at least 1 sample
                
                if len(pattern_list) > samples_per_ticker:
                    pattern_list = np.random.choice(
                        pattern_list, 
                        size=samples_per_ticker, 
                        replace=False
                    ).tolist()
                
                # Generate chart for each pattern instance
                for pattern in pattern_list:
                    try:
                        # Extract pattern indices based on pattern type
                        if pattern_name == 'Head and shoulders top':
                            start_idx = min(pattern['left_shoulder'], pattern['troughs'][0])
                            end_idx = max(pattern['right_shoulder'], pattern['troughs'][-1])
                            highlight_points = [
                                pattern['left_shoulder'], 
                                pattern['head'], 
                                pattern['right_shoulder']
                            ] + pattern['troughs']
                            
                        elif pattern_name == 'Head and shoulders bottom':
                            start_idx = min(pattern['left_shoulder'], pattern['peaks'][0])
                            end_idx = max(pattern['right_shoulder'], pattern['peaks'][-1])
                            highlight_points = [
                                pattern['left_shoulder'], 
                                pattern['head'], 
                                pattern['right_shoulder']
                            ] + pattern['peaks']
                            
                        elif pattern_name == 'M_Head':
                            start_idx = pattern['first_top']
                            end_idx = pattern['confirmation_trough']
                            highlight_points = [
                                pattern['first_top'], 
                                pattern['middle_trough'],
                                pattern['second_top'],
                                pattern['confirmation_trough']
                            ]
                            
                        elif pattern_name == 'W_Bottom':
                            start_idx = pattern['first_bottom']
                            end_idx = pattern['confirmation_peak']
                            highlight_points = [
                                pattern['first_bottom'], 
                                pattern['middle_peak'],
                                pattern['second_bottom'],
                                pattern['confirmation_peak']
                            ]
                            
                        elif pattern_name == 'Triangle':
                            start_idx = pattern['start_index']
                            end_idx = pattern['end_index']
                            highlight_points = pattern['high_points'] + pattern['low_points']
                            
                        elif pattern_name == 'StockLine':
                            start_idx = pattern['start_index']
                            end_idx = pattern['end_index']
                            highlight_points = list(range(start_idx, end_idx + 1))
                        
                        # Add buffer for context (10 candles before, 5 after if possible)
                        context_before = 10
                        context_after = 5
                        
                        buffer_start = max(0, start_idx - context_before)
                        buffer_end = min(len(data) - 1, end_idx + context_after)
                        
                        # Skip if pattern is too close to beginning or end of data
                        if buffer_start >= buffer_end or (buffer_end - buffer_start) < 15:
                            continue
                        
                        # Extract window data
                        window_data = data.iloc[buffer_start:buffer_end+1].copy()
                        window_data.reset_index(inplace=True)
                        
                        # Create figure
                        fig_num = len(generated_charts) % 20  # Reuse figure numbers to avoid too many open figures
                        plt.close(fig_num)  # Close if previously opened
                        fig = plt.figure(num=fig_num, figsize=(12, 8), dpi=100)
                        ax = fig.add_subplot(1, 1, 1)
                        
                        # Plot candlestick chart
                        for i, (_, candle) in enumerate(window_data.iterrows()):
                            # Original index in the full dataset
                            orig_idx = buffer_start + i
                            
                            # Determine if this candle is part of the pattern
                            is_pattern = orig_idx in highlight_points
                            
                            # Calculate colors based on price movement
                            if candle['Close'] >= candle['Open']:
                                color = 'green'
                                body_height = candle['Close'] - candle['Open']
                            else:
                                color = 'red'
                                body_height = candle['Open'] - candle['Close']
                            
                            # Width of the candlestick
                            width = 0.6
                            
                            # Plot the candlestick body
                            rect = Rectangle(
                                xy=(i-width/2, min(candle['Open'], candle['Close'])),
                                width=width,
                                height=body_height,
                                facecolor=color,
                                edgecolor='black' if not is_pattern else 'blue',
                                linewidth=0.5 if not is_pattern else 2.0
                            )
                            ax.add_patch(rect)
                            
                            # Plot the upper and lower wicks
                            ax.plot([i, i], [candle['Low'], min(candle['Open'], candle['Close'])], 
                                   color='black' if not is_pattern else 'blue', 
                                   linewidth=0.5 if not is_pattern else 2.0)
                            ax.plot([i, i], [max(candle['Open'], candle['Close']), candle['High']], 
                                   color='black' if not is_pattern else 'blue', 
                                   linewidth=0.5 if not is_pattern else 2.0)
                        
                        # For StockLine, draw the horizontal line
                        if pattern_name == 'StockLine':
                            price_level = pattern['price_level']
                            ax.axhline(y=price_level, color='blue', linestyle='--', linewidth=2.0)
                        
                        # For Triangle, draw the trend lines
                        if pattern_name == 'Triangle':
                            # Adjust indices for the window
                            high_points_adj = [p - buffer_start for p in pattern['high_points']]
                            low_points_adj = [p - buffer_start for p in pattern['low_points']]
                            
                            # Get the high and low prices
                            high_prices = [window_data.iloc[i]['High'] for i in high_points_adj]
                            low_prices = [window_data.iloc[i]['Low'] for i in low_points_adj]
                            
                            # Plot trend lines if there are enough points
                            if len(high_points_adj) >= 2:
                                high_indices = range(len(high_points_adj))
                                high_coef = np.polyfit(high_indices, high_prices, 1)
                                high_poly = np.poly1d(high_coef)
                                plt.plot(high_points_adj, high_poly(high_indices), color='blue', linestyle='--', linewidth=2.0)
                            
                            if len(low_points_adj) >= 2:
                                low_indices = range(len(low_points_adj))
                                low_coef = np.polyfit(low_indices, low_prices, 1)
                                low_poly = np.poly1d(low_coef)
                                plt.plot(low_points_adj, low_poly(low_indices), color='blue', linestyle='--', linewidth=2.0)
                        
                        # Set y-axis limits with some padding
                        y_min = window_data['Low'].min() * 0.95
                        y_max = window_data['High'].max() * 1.05
                        ax.set_ylim(y_min, y_max)
                        
                        # Set x-axis to show dates
                        ax.set_xlim(-0.5, len(window_data) - 0.5)
                        
                        # Add labels and title
                        ax.set_title(f"{ticker} - {pattern_name} Pattern")
                        ax.set_ylabel("Price")
                        ax.set_xlabel("Candle Index")
                        
                        # Generate a unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        rand_id = np.random.randint(0, 10000)
                        pattern_folder = pattern_name.replace(' ', '_')
                        chart_path = os.path.join(
                            output_dir, 
                            pattern_folder, 
                            f"{ticker}_{pattern_folder}_{timestamp}_{rand_id}.jpg"
                        )
                        
                        # Calculate pattern bounding box for annotation
                        # Normalize pattern indices to the window
                        if pattern_name == 'Head and shoulders top' or pattern_name == 'Head and shoulders bottom':
                            # For head and shoulders, use the three peaks/troughs and neckline
                            x_points = [p - buffer_start for p in [pattern['left_shoulder'], pattern['head'], pattern['right_shoulder']]]
                            y_points = []
                            for p in [pattern['left_shoulder'], pattern['head'], pattern['right_shoulder']]:
                                if p < len(data):
                                    y_points.append(data.iloc[p]['Close'])
                            
                            x_min, x_max = min(x_points), max(x_points)
                            y_min, y_max = min(y_points), max(y_points)
                            
                            # Include neckline
                            y_min = min(y_min, pattern['neckline'])
                            
                        elif pattern_name == 'M_Head':
                            # For M-Head, use the four points of the M
                            x_points = [p - buffer_start for p in [pattern['first_top'], pattern['middle_trough'], 
                                                                 pattern['second_top'], pattern['confirmation_trough']]]
                            y_points = []
                            for p in [pattern['first_top'], pattern['middle_trough'], pattern['second_top'], pattern['confirmation_trough']]:
                                if p < len(data):
                                    y_points.append(data.iloc[p]['Close'])
                            
                            x_min, x_max = min(x_points), max(x_points)
                            y_min, y_max = min(y_points), max(y_points)
                            
                        elif pattern_name == 'W_Bottom':
                            # For W-Bottom, use the four points of the W
                            x_points = [p - buffer_start for p in [pattern['first_bottom'], pattern['middle_peak'], 
                                                                 pattern['second_bottom'], pattern['confirmation_peak']]]
                            y_points = []
                            for p in [pattern['first_bottom'], pattern['middle_peak'], pattern['second_bottom'], pattern['confirmation_peak']]:
                                if p < len(data):
                                    y_points.append(data.iloc[p]['Close'])
                            
                            x_min, x_max = min(x_points), max(x_points)
                            y_min, y_max = min(y_points), max(y_points)
                            
                        elif pattern_name == 'Triangle':
                            # For Triangle, use the high and low points
                            high_points_adj = [p - buffer_start for p in pattern['high_points']]
                            low_points_adj = [p - buffer_start for p in pattern['low_points']]
                            
                            x_points = high_points_adj + low_points_adj
                            y_points = []
                            
                            for p in pattern['high_points']:
                                if p < len(data):
                                    y_points.append(data.iloc[p]['High'])
                            
                            for p in pattern['low_points']:
                                if p < len(data):
                                    y_points.append(data.iloc[p]['Low'])
                            
                            x_min, x_max = min(x_points), max(x_points)
                            y_min, y_max = min(y_points), max(y_points)
                            
                        elif pattern_name == 'StockLine':
                            # For StockLine, use the range and price level
                            x_min = pattern['start_index'] - buffer_start
                            x_max = pattern['end_index'] - buffer_start
                            
                            price_level = pattern['price_level']
                            price_range = price_level * 0.02  # 2% of price level
                            
                            y_min = price_level - price_range
                            y_max = price_level + price_range
                        
                        # Ensure x_min and x_max are within the chart
                        x_min = max(0, x_min)
                        x_max = min(len(window_data) - 1, x_max)
                        
                        # Save the chart as an image
                        plt.tight_layout()
                        plt.savefig(chart_path)
                        plt.close(fig)
                        
                        # Calculate YOLO format annotations
                        # Convert to YOLO format: [class_id, x_center, y_center, width, height]
                        img_width = fig.get_size_inches()[0] * fig.dpi
                        img_height = fig.get_size_inches()[1] * fig.dpi
                        
                        # All values normalized to 0-1
                        class_id = pattern_types.index(pattern_name)
                        
                        # Calculate bounding box in YOLO format
                        # x_center and y_center are normalized coordinates in the image
                        # Normalize x-coordinates to image width based on number of candles
                        x_center = ((x_min + x_max) / 2) / len(window_data)
                        y_center = ((y_min + y_max) / 2 - y_min) / (y_max - y_min)
                        
                        # Width and height are normalized by image dimensions
                        width_norm = (x_max - x_min) / len(window_data)
                        height_norm = (y_max - y_min) / (y_max - y_min)
                        
                        # Create YOLO annotation file
                        annotation_path = chart_path.replace('.jpg', '.txt')
                        with open(annotation_path, 'w') as f:
                            f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}")
                        
                        generated_charts.append(chart_path)
                        print(f"Generated chart for {pattern_name} pattern in {ticker}")
                        
                    except Exception as e:
                        print(f"Error generating chart for {pattern_name} in {ticker}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Print summary of pattern counts
    print("\nPattern detection summary:")
    for pattern_name, count in pattern_counts.items():
        print(f"{pattern_name}: {count}")
    
    return generated_charts


def detect_head_and_shoulders_top(data, peaks, troughs, threshold=0.03):
    """
    Detect head and shoulders top pattern.
    
    A head and shoulders top has:
    - Three peaks with the middle one (head) higher than the other two (shoulders)
    - The two shoulders should be of similar height
    - Four troughs, with the middle two being at similar levels
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    peaks (array): Indices of price peaks
    troughs (array): Indices of price troughs
    threshold (float): Tolerance for shoulder height similarity
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if peaks is None or troughs is None:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    for i in range(len(peaks) - 2):
        # Get three consecutive peaks (potential left shoulder, head, right shoulder)
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        
        # Check if the middle peak (head) is higher than the two surrounding peaks (shoulders)
        if prices[p2] > prices[p1] and prices[p2] > prices[p3]:
            # Check if the two shoulders are of similar height
            shoulder_diff = abs(prices[p1] - prices[p3]) / prices[p1]
            if shoulder_diff <= threshold:
                # Find the troughs between the peaks
                troughs_between = [t for t in troughs if p1 < t < p3]
                if len(troughs_between) >= 2:
                    # Get the neckline (connecting the troughs)
                    t1 = max([t for t in troughs if t < p1], default=0)
                    t2 = min([t for t in troughs if t > p1 and t < p2], default=p2)
                    t3 = min([t for t in troughs if t > p2 and t < p3], default=p3)
                    t4 = min([t for t in troughs if t > p3], default=len(prices)-1)
                    
                    if t1 < p1 < t2 < p2 < t3 < p3 < t4:
                        # Calculate neckline as the average of the middle troughs
                        neckline = (prices[t2] + prices[t3]) / 2
                        
                        # Check if the pattern is complete (price breaks below neckline after right shoulder)
                        if t4 < len(prices) and prices[t4] < neckline:
                            patterns.append({
                                'type': 'Head and shoulders top',
                                'left_shoulder': p1,
                                'head': p2,
                                'right_shoulder': p3,
                                'troughs': [t1, t2, t3, t4],
                                'neckline': neckline,
                                'confidence': 1 - shoulder_diff  # Higher when shoulders are more similar
                            })
    
    return patterns
    
def detect_head_and_shoulders_bottom(data, peaks, troughs, threshold=0.03):
    """
    Detect head and shoulders bottom (inverse head and shoulders) pattern.
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    peaks (array): Indices of price peaks
    troughs (array): Indices of price troughs
    threshold (float): Tolerance for shoulder height similarity
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if peaks is None or troughs is None:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    for i in range(len(troughs) - 2):
        # Get three consecutive troughs (potential left shoulder, head, right shoulder)
        t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
        
        # Check if the middle trough (head) is lower than the two surrounding troughs (shoulders)
        if prices[t2] < prices[t1] and prices[t2] < prices[t3]:
            # Check if the two shoulders are of similar height
            shoulder_diff = abs(prices[t1] - prices[t3]) / prices[t1]
            if shoulder_diff <= threshold:
                # Find the peaks between the troughs
                peaks_between = [p for p in peaks if t1 < p < t3]
                if len(peaks_between) >= 2:
                    # Get the neckline (connecting the peaks)
                    p1 = max([p for p in peaks if p < t1], default=0)
                    p2 = min([p for p in peaks if p > t1 and p < t2], default=t2)
                    p3 = min([p for p in peaks if p > t2 and p < t3], default=t3)
                    p4 = min([p for p in peaks if p > t3], default=len(prices)-1)
                    
                    if p1 < t1 < p2 < t2 < p3 < t3 < p4:
                        # Calculate neckline as the average of the middle peaks
                        neckline = (prices[p2] + prices[p3]) / 2
                        
                        # Check if the pattern is complete (price breaks above neckline after right shoulder)
                        if p4 < len(prices) and prices[p4] > neckline:
                            patterns.append({
                                'type': 'Head and shoulders bottom',
                                'left_shoulder': t1,
                                'head': t2,
                                'right_shoulder': t3,
                                'peaks': [p1, p2, p3, p4],
                                'neckline': neckline,
                                'confidence': 1 - shoulder_diff  # Higher when shoulders are more similar
                            })
    
    return patterns

def detect_w_bottom(data, peaks, troughs, threshold=0.05):
    """
    Detect W bottom (double bottom) pattern.
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    peaks (array): Indices of price peaks
    troughs (array): Indices of price troughs
    threshold (float): Tolerance for bottom similarity
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if peaks is None or troughs is None:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    for i in range(len(troughs) - 1):
        # Get two consecutive troughs
        t1, t2 = troughs[i], troughs[i+1]
        
        # Check if they are at similar levels (bottoms of the W)
        bottom_diff = abs(prices[t1] - prices[t2]) / prices[t1]
        if bottom_diff <= threshold:
            # Find the peak between the troughs
            peaks_between = [p for p in peaks if t1 < p < t2]
            if len(peaks_between) >= 1:
                middle_peak = peaks_between[0]
                
                # Find the peak after the second trough
                peaks_after = [p for p in peaks if p > t2]
                if len(peaks_after) >= 1:
                    confirmation_peak = peaks_after[0]
                    
                    # Check if the confirmation peak is higher than the middle peak
                    if prices[confirmation_peak] > prices[middle_peak]:
                        patterns.append({
                            'type': 'W_Bottom',
                            'first_bottom': t1,
                            'second_bottom': t2,
                            'middle_peak': middle_peak,
                            'confirmation_peak': confirmation_peak,
                            'confidence': 1 - bottom_diff  # Higher when bottoms are more similar
                        })
    
    return patterns

def detect_m_head(data, peaks, troughs, threshold=0.05):
    """
    Detect M top (double top) pattern.
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    peaks (array): Indices of price peaks
    troughs (array): Indices of price troughs
    threshold (float): Tolerance for top similarity
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if peaks is None or troughs is None:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    for i in range(len(peaks) - 1):
        # Get two consecutive peaks
        p1, p2 = peaks[i], peaks[i+1]
        
        # Check if they are at similar levels (tops of the M)
        top_diff = abs(prices[p1] - prices[p2]) / prices[p1]
        if top_diff <= threshold:
            # Find the trough between the peaks
            troughs_between = [t for t in troughs if p1 < t < p2]
            if len(troughs_between) >= 1:
                middle_trough = troughs_between[0]
                
                # Find the trough after the second peak
                troughs_after = [t for t in troughs if t > p2]
                if len(troughs_after) >= 1:
                    confirmation_trough = troughs_after[0]
                    
                    # Check if the confirmation trough is lower than the middle trough
                    if prices[confirmation_trough] < prices[middle_trough]:
                        patterns.append({
                            'type': 'M_Head',
                            'first_top': p1,
                            'second_top': p2,
                            'middle_trough': middle_trough,
                            'confirmation_trough': confirmation_trough,
                            'confidence': 1 - top_diff  # Higher when tops are more similar
                        })
    
    return patterns

def detect_triangle(data, peaks, troughs, min_points=5):
    """
    Detect triangle patterns (ascending, descending, and symmetric).
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    peaks (array): Indices of price peaks
    troughs (array): Indices of price troughs
    min_points (int): Minimum number of points to form a triangle
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if peaks is None or troughs is None:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    # Combine peaks and troughs and sort by index
    extrema = sorted([(i, prices[i], 'peak') for i in peaks] + 
                    [(i, prices[i], 'trough') for i in troughs], 
                    key=lambda x: x[0])
    
    if len(extrema) < min_points:
        return []
        
    # Analyze segments of the price series
    for i in range(len(extrema) - min_points + 1):
        segment = extrema[i:i+min_points]
        segment_indices = [x[0] for x in segment]
        segment_prices = [x[1] for x in segment]
        segment_types = [x[2] for x in segment]
        
        # Extract high points and low points
        high_points = [(idx, price) for idx, (idx, price, type_) in enumerate(segment) if type_ == 'peak']
        low_points = [(idx, price) for idx, (idx, price, type_) in enumerate(segment) if type_ == 'trough']
        
        if len(high_points) < 2 or len(low_points) < 2:
            continue
            
        # Calculate linear regression for high points and low points
        high_indices = [hp[0] for hp in high_points]
        high_prices = [hp[1] for hp in high_points]
        
        low_indices = [lp[0] for lp in low_points]
        low_prices = [lp[1] for lp in low_points]
        
        # Calculate slopes of upper and lower boundaries
        if len(high_indices) >= 2:
            high_slope = np.polyfit(high_indices, high_prices, 1)[0]
        else:
            high_slope = 0
            
        if len(low_indices) >= 2:
            low_slope = np.polyfit(low_indices, low_prices, 1)[0]
        else:
            low_slope = 0
        
        # Determine triangle type based on slopes
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            triangle_type = "Ascending"
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            triangle_type = "Descending"
        elif high_slope < -0.001 and low_slope > 0.001:
            triangle_type = "Symmetric"
        else:
            continue  # Not a triangle pattern
            
        # Calculate convergence point
        if high_slope != low_slope:
            x_intersect = (np.polyfit(low_indices, low_prices, 1)[1] - np.polyfit(high_indices, high_prices, 1)[1]) / (high_slope - low_slope)
            if x_intersect > max(segment_indices):
                patterns.append({
                    'type': 'Triangle',
                    'subtype': triangle_type,
                    'start_index': segment_indices[0],
                    'end_index': segment_indices[-1],
                    'high_points': [segment_indices[hp[0]] for hp in high_points],
                    'low_points': [segment_indices[lp[0]] for lp in low_points],
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'convergence_point': x_intersect
                })
                
    return patterns

def detect_stock_line(data, window_size=20, threshold=0.02):
    """
    Detect horizontal support/resistance lines (stock lines).
    
    Parameters:
    data (pandas.DataFrame): Stock price data with OHLC columns
    window_size (int): Size of the window to analyze
    threshold (float): Maximum allowed deviation from the line
    
    Returns:
    list: List of dictionaries with detected pattern details
    """
    if data is None or len(data) == 0:
        return []
        
    patterns = []
    prices = data['Close'].values
    
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i+window_size]
        avg_price = np.mean(window)
        max_deviation = np.max(np.abs(window - avg_price) / avg_price)
        
        if max_deviation <= threshold:
            # Count how many times the price touches the line
            touches = sum(1 for p in window if abs(p - avg_price) / avg_price <= threshold / 2)
            
            if touches >= 3:  # At least 3 touches to be significant
                patterns.append({
                    'type': 'StockLine',
                    'subtype': 'Support/Resistance',
                    'start_index': i,
                    'end_index': i + window_size - 1,
                    'price_level': avg_price,
                    'touches': touches,
                    'confidence': 1 - max_deviation/threshold  # Higher when deviations are smaller
                })
                
    return patterns

# Example usage
if __name__ == "__main__":
    # Example usage with a list of tickers
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    generated_charts = generate_pattern_dataset(tickers, num_samples_per_pattern=50)
    print(f"Generated {len(generated_charts)} chart images with pattern annotations")
