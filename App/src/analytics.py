#!/usr/bin/env python3
"""
Energy Data Analytics
Comprehensive analytics for multiple data granularities:
- Long-term trends (monthly data, 24 months)
- Medium-term patterns (hourly data, 1 month)
- Short-term volatility (5-minute data, 1 week)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, Optional

class DualMetricAnalytics:
    """Analytics for both power and market value data with side-by-side plotting"""
    
    def __init__(self, power_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]):
        """Initialize with both power and market value data"""
        self.power_data = power_data
        self.market_data = market_data
        self.network = None
        
        # Extract network info from first available dataset
        for data_dict in [power_data, market_data]:
            for dataset_key, df in data_dict.items():
                if not df.empty:
                    self.network = df['network'].iloc[0]
                    break
            if self.network:
                break
    
    def create_dual_visualizations(self, save_path: str = None) -> None:
        """Create combined line plots showing both power and market value in same plots"""
        print("📊 Creating combined visualizations...")
        print("   └─ Generating dual-axis line plots...")
        
        fig = plt.figure(figsize=(18, 16))  # Increased height further
        fig.suptitle(f'{self.network} - Power vs Market Value Combined Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)  # Adjusted title position
        
        plots_created = 0
        
        # Monthly combined plot (24 months)
        if ('monthly_24months' in self.power_data and not self.power_data['monthly_24months'].empty and
            'monthly_24months' in self.market_data and not self.market_data['monthly_24months'].empty):
            
            print("   📈 Creating monthly trends plot (24 months)...")
            power_df = self.power_data['monthly_24months']
            market_df = self.market_data['monthly_24months']
            
            ax1 = plt.subplot(2, 2, 1)
            
            # Create twin axis for different units
            ax1_twin = ax1.twinx()
            
            # Plot power on left axis
            line1 = ax1.plot(power_df['timestamp'], power_df['value'], 
                            linewidth=2.5, marker='o', markersize=5, color='#2E86AB', 
                            label='Power Generation', alpha=0.8)
            ax1.set_ylabel(f"Power ({power_df['unit'].iloc[0]})", color='#2E86AB', fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#2E86AB')
            
            # Plot market value on right axis
            line2 = ax1_twin.plot(market_df['timestamp'], market_df['value'], 
                                 linewidth=2.5, marker='s', markersize=5, color='#A23B72', 
                                 label='Market Value', alpha=0.8)
            ax1_twin.set_ylabel(f"Market Value ({market_df['unit'].iloc[0]})", color='#A23B72', fontweight='bold')
            ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
            
            ax1.set_title('Monthly Trends: Power Generation vs Market Value\n(Oct 2023 - Sep 2025)', 
                         fontsize=11, fontweight='bold', pad=8)  # Reduced gap from 15 to 8
            ax1.tick_params(axis='x', rotation=30, labelsize=9)  # 30 degree rotation
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
            plots_created += 1
        
        # Hourly pattern combined plot (1 month)
        if ('hourly_1month' in self.power_data and not self.power_data['hourly_1month'].empty and
            'hourly_1month' in self.market_data and not self.market_data['hourly_1month'].empty):
            
            print("   ⏰ Creating hourly patterns plot (1 month)...")
            power_df = self.power_data['hourly_1month'].copy()
            market_df = self.market_data['hourly_1month'].copy()
            
            power_df['hour'] = power_df['timestamp'].dt.hour
            market_df['hour'] = market_df['timestamp'].dt.hour
            
            power_hourly_avg = power_df.groupby('hour')['value'].mean()
            market_hourly_avg = market_df.groupby('hour')['value'].mean()
            
            ax2 = plt.subplot(2, 2, 2)
            ax2_twin = ax2.twinx()
            
            # Plot power hourly pattern
            line1 = ax2.plot(power_hourly_avg.index, power_hourly_avg.values, 
                            linewidth=2.5, marker='o', markersize=5, color='#2E86AB', 
                            label='Power Generation', alpha=0.8)
            ax2.set_ylabel(f"Power ({power_df['unit'].iloc[0]})", color='#2E86AB', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#2E86AB')
            
            # Plot market value hourly pattern
            line2 = ax2_twin.plot(market_hourly_avg.index, market_hourly_avg.values, 
                                 linewidth=2.5, marker='s', markersize=5, color='#A23B72', 
                                 label='Market Value', alpha=0.8)
            ax2_twin.set_ylabel(f"Market Value ({market_df['unit'].iloc[0]})", color='#A23B72', fontweight='bold')
            ax2_twin.tick_params(axis='y', labelcolor='#A23B72')
            
            ax2.set_title('Daily Patterns: Average Hourly Demand vs Pricing\n(Past 30 Days)', 
                         fontsize=11, fontweight='bold', pad=8)  # Reduced gap from 15 to 8
            ax2.set_xlabel('Hour of Day (0-23)', fontweight='bold', fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xticks(range(0, 24, 3))
            ax2.tick_params(labelsize=9)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
            plots_created += 1
        
        # Hourly time series combined plot (1 month)
        if ('hourly_1month' in self.power_data and not self.power_data['hourly_1month'].empty and
            'hourly_1month' in self.market_data and not self.market_data['hourly_1month'].empty):
            
            print("   📊 Creating hourly time series plot (1 month)...")
            power_df = self.power_data['hourly_1month']
            market_df = self.market_data['hourly_1month']
            
            ax3 = plt.subplot(2, 2, 3)
            ax3_twin = ax3.twinx()
            
            # Plot power time series
            line1 = ax3.plot(power_df['timestamp'], power_df['value'], 
                            linewidth=1.5, alpha=0.8, color='#2E86AB', label='Power Generation')
            ax3.set_ylabel(f"Power ({power_df['unit'].iloc[0]})", color='#2E86AB', fontweight='bold')
            ax3.tick_params(axis='y', labelcolor='#2E86AB')
            
            # Plot market value time series
            line2 = ax3_twin.plot(market_df['timestamp'], market_df['value'], 
                                 linewidth=1.5, alpha=0.8, color='#A23B72', label='Market Value')
            ax3_twin.set_ylabel(f"Market Value ({market_df['unit'].iloc[0]})", color='#A23B72', fontweight='bold')
            ax3_twin.tick_params(axis='y', labelcolor='#A23B72')
            
            ax3.set_title('Hourly Time Series: Generation vs Market Response\n(Past 30 Days)', 
                         fontsize=11, fontweight='bold', pad=8)  # Reduced gap from 15 to 8
            ax3.tick_params(axis='x', rotation=30, labelsize=9)  # 30 degree rotation
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
            plots_created += 1
        
        # 5-minute combined plot (1 week)
        if ('fivemin_1week' in self.power_data and not self.power_data['fivemin_1week'].empty and
            'fivemin_1week' in self.market_data and not self.market_data['fivemin_1week'].empty):
            
            print("   ⚡ Creating 5-minute real-time plot (1 week)...")
            power_df = self.power_data['fivemin_1week']
            market_df = self.market_data['fivemin_1week']
            
            ax4 = plt.subplot(2, 2, 4)
            ax4_twin = ax4.twinx()
            
            # Plot power 5-minute data
            line1 = ax4.plot(power_df['timestamp'], power_df['value'], 
                            linewidth=1, alpha=0.9, color='#2E86AB', label='Power Generation')
            ax4.set_ylabel(f"Power ({power_df['unit'].iloc[0]})", color='#2E86AB', fontweight='bold')
            ax4.tick_params(axis='y', labelcolor='#2E86AB')
            
            # Plot market value 5-minute data
            line2 = ax4_twin.plot(market_df['timestamp'], market_df['value'], 
                                 linewidth=1, alpha=0.9, color='#A23B72', label='Market Value')
            ax4_twin.set_ylabel(f"Market Value ({market_df['unit'].iloc[0]})", color='#A23B72', fontweight='bold')
            ax4_twin.tick_params(axis='y', labelcolor='#A23B72')
            
            ax4.set_title('Real-time Analysis: 5-Minute Market Dynamics\n(Past 7 Days)', 
                         fontsize=11, fontweight='bold', pad=8)  # Reduced gap from 15 to 8
            ax4.tick_params(axis='x', rotation=30, labelsize=9)  # 30 degree rotation
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
            plots_created += 1
        
        # Adjust layout with better spacing
        plt.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=6.0, w_pad=2.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"   ✅ Visualizations saved: {save_path}")
        
        print(f"   📊 Generated {plots_created} combined plots successfully!")
        plt.show()
    
    def generate_dual_report(self) -> str:
        """Generate comprehensive report for both metrics"""
        power_analytics = EnergyAnalytics(self.power_data)
        market_analytics = EnergyAnalytics(self.market_data)
        
        power_analysis = power_analytics.analyze_all()
        market_analysis = market_analytics.analyze_all()
        
        report = f"""
DUAL ENERGY ANALYTICS REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

POWER DATA ANALYSIS
{'-' * 30}
Network: {power_analysis['summary']['network']}
Metric: {power_analysis['summary']['metric']}
Unit: {power_analysis['summary']['unit']}
"""
        
        # Power datasets summary
        for dataset_key, stats in power_analysis['summary']['datasets'].items():
            report += f"""
{dataset_key.upper().replace('_', ' ')}:
  Records: {stats['records']:,}
  Period: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}
  Range: {stats['value_stats']['min']:,.0f} - {stats['value_stats']['max']:,.0f} {power_analysis['summary']['unit']}
  Average: {stats['value_stats']['mean']:,.0f} {power_analysis['summary']['unit']}
"""
        
        # Power insights
        if 'error' not in power_analysis['long_term']:
            lt = power_analysis['long_term']
            report += f"""
POWER INSIGHTS:
• Trend Direction: {lt['trend_direction'].title()}
• Peak Month: {lt['peak_month']} | Low Month: {lt['low_month']}
"""
        
        report += f"""

MARKET VALUE DATA ANALYSIS
{'-' * 30}
Network: {market_analysis['summary']['network']}
Metric: {market_analysis['summary']['metric']}
Unit: {market_analysis['summary']['unit']}
"""
        
        # Market Value datasets summary
        for dataset_key, stats in market_analysis['summary']['datasets'].items():
            report += f"""
{dataset_key.upper().replace('_', ' ')}:
  Records: {stats['records']:,}
  Period: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}
  Range: {stats['value_stats']['min']:,.2f} - {stats['value_stats']['max']:,.2f} {market_analysis['summary']['unit']}
  Average: {stats['value_stats']['mean']:,.2f} {market_analysis['summary']['unit']}
"""
        
        # Market Value insights
        if 'error' not in market_analysis['long_term']:
            lt = market_analysis['long_term']
            report += f"""
MARKET VALUE INSIGHTS:
• Trend Direction: {lt['trend_direction'].title()}
• Peak Month: {lt['peak_month']} | Low Month: {lt['low_month']}
"""
        
        return report


class EnergyAnalytics:
    """Comprehensive analytics for energy data"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """Initialize with data dictionary containing different granularities"""
        self.data = data
        self.network = None
        self.metric = None
        self.unit = None
        
        # Extract metadata from first available dataset
        for dataset_key, df in data.items():
            if not df.empty:
                self.network = df['network'].iloc[0]
                self.metric = df['metric'].iloc[0]
                self.unit = df['unit'].iloc[0]
                break
    
    def analyze_all(self) -> Dict:
        """Run all analysis and return comprehensive results"""
        results = {
            'summary': self.get_summary_stats(),
            'long_term': self.analyze_long_term_trends(),
            'medium_term': self.analyze_medium_term_patterns(),
            'short_term': self.analyze_short_term_volatility()
        }
        return results
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for all datasets"""
        summary = {
            'network': self.network,
            'metric': self.metric,
            'unit': self.unit,
            'datasets': {}
        }
        
        for dataset_key, df in self.data.items():
            if df.empty:
                continue
                
            stats = {
                'records': len(df),
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'value_stats': {
                    'min': df['value'].min(),
                    'max': df['value'].max(),
                    'mean': df['value'].mean(),
                    'median': df['value'].median(),
                    'std': df['value'].std()
                }
            }
            summary['datasets'][dataset_key] = stats
        
        return summary
    
    def analyze_long_term_trends(self) -> Dict:
        """Analyze long-term trends using monthly data"""
        if 'monthly_24months' not in self.data or self.data['monthly_24months'].empty:
            return {'error': 'No monthly data available'}
        
        df = self.data['monthly_24months'].copy()
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        
        # Trend analysis
        x = np.arange(len(df))
        coeffs = np.polyfit(x, df['value'], 1)
        trend_slope = coeffs[0]
        
        # Seasonal patterns
        monthly_avg = df.groupby('month')['value'].mean()
        
        # Year-over-year comparison
        yearly_stats = df.groupby('year')['value'].mean()
        yoy_change = None
        if len(yearly_stats) >= 2:
            years = list(yearly_stats.index)
            current = yearly_stats.iloc[-1]
            previous = yearly_stats.iloc[-2]
            yoy_change = ((current - previous) / previous) * 100
        
        return {
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
            'trend_slope': trend_slope,
            'peak_month': monthly_avg.idxmax(),
            'low_month': monthly_avg.idxmin(),
            'year_over_year_change': yoy_change,
            'overall_volatility': df['value'].std() / df['value'].mean()
        }
    
    def analyze_medium_term_patterns(self) -> Dict:
        """Analyze medium-term patterns using hourly data"""
        if 'hourly_1month' not in self.data or self.data['hourly_1month'].empty:
            return {'error': 'No hourly data available'}
        
        df = self.data['hourly_1month'].copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        
        # Hourly patterns
        hourly_avg = df.groupby('hour')['value'].mean()
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        # Weekly patterns
        weekly_avg = df.groupby('day_of_week')['value'].mean()
        
        # Weekend vs weekday
        weekend_avg = df[df['is_weekend']]['value'].mean()
        weekday_avg = df[~df['is_weekend']]['value'].mean()
        weekend_premium = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        
        # Load factor
        load_factor = df['value'].mean() / df['value'].max()
        
        return {
            'peak_hour': peak_hour,
            'low_hour': low_hour,
            'peak_day': weekly_avg.idxmax(),
            'weekend_premium': weekend_premium,
            'load_factor': load_factor,
            'hourly_variation': hourly_avg.max() - hourly_avg.min()
        }
    
    def analyze_short_term_volatility(self) -> Dict:
        """Analyze short-term volatility using 5-minute data"""
        if 'fivemin_1week' not in self.data or self.data['fivemin_1week'].empty:
            return {'error': 'No 5-minute data available'}
        
        df = self.data['fivemin_1week'].copy()
        
        # Price spike detection
        mean_val = df['value'].mean()
        std_val = df['value'].std()
        spike_threshold = mean_val + (2.5 * std_val)
        spikes = df[df['value'] > spike_threshold]
        
        # Rapid changes
        df['pct_change'] = df['value'].pct_change() * 100
        rapid_changes = df[abs(df['pct_change']) > 5.0]
        
        # Stability metrics
        stability_index = 1 / (1 + std_val / mean_val)
        
        # Intraday volatility
        df['date'] = df['timestamp'].dt.date
        daily_vol = df.groupby('date')['value'].std().mean()
        
        return {
            'price_spikes': len(spikes),
            'spike_frequency': len(spikes) / len(df) * 100,
            'rapid_changes': len(rapid_changes),
            'stability_index': stability_index,
            'avg_daily_volatility': daily_vol,
            'max_5min_change': abs(df['pct_change']).max()
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive text report"""
        analysis = self.analyze_all()
        
        report = f"""
ENERGY ANALYTICS REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
{'-' * 20}
Network: {analysis['summary']['network']}
Metric: {analysis['summary']['metric']}
Unit: {analysis['summary']['unit']}

DATASETS
{'-' * 20}
"""
        
        for dataset_key, stats in analysis['summary']['datasets'].items():
            report += f"""
{dataset_key.upper().replace('_', ' ')}:
  Records: {stats['records']:,}
  Period: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}
  Range: {stats['value_stats']['min']:,.0f} - {stats['value_stats']['max']:,.0f} {analysis['summary']['unit']}
  Average: {stats['value_stats']['mean']:,.0f} {analysis['summary']['unit']}
"""
        
        if 'error' not in analysis['long_term']:
            lt = analysis['long_term']
            report += f"""
LONG-TERM TRENDS (24 Months)
{'-' * 30}
• Trend Direction: {lt['trend_direction'].title()}
• Peak Month: {lt['peak_month']} | Low Month: {lt['low_month']}
• Overall Volatility: {lt['overall_volatility']:.3f}
"""
            if lt['year_over_year_change']:
                report += f"• Year-over-Year Change: {lt['year_over_year_change']:+.1f}%\n"
        
        if 'error' not in analysis['medium_term']:
            mt = analysis['medium_term']
            report += f"""
MEDIUM-TERM PATTERNS (1 Month)
{'-' * 30}
• Peak Hour: {mt['peak_hour']:02d}:00 | Low Hour: {mt['low_hour']:02d}:00
• Peak Day: {mt['peak_day']}
• Weekend Premium: {mt['weekend_premium']:+.1f}%
• Load Factor: {mt['load_factor']:.3f}
"""
        
        if 'error' not in analysis['short_term']:
            st = analysis['short_term']
            report += f"""
SHORT-TERM VOLATILITY (1 Week)
{'-' * 30}
• Price Spikes: {st['price_spikes']} ({st['spike_frequency']:.2f}%)
• Rapid Changes (>5%): {st['rapid_changes']}
• Stability Index: {st['stability_index']:.3f}
• Max 5-min Change: {st['max_5min_change']:.2f}%
"""
        
        return report
    
    def create_visualizations(self, save_path: str = None) -> None:
        """Create line plot visualizations only"""
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{self.network} {self.metric} - Line Plot Analysis', fontsize=16)
        
        plot_count = 0
        
        # Long-term plot (monthly data)
        if 'monthly_24months' in self.data and not self.data['monthly_24months'].empty:
            df = self.data['monthly_24months']
            ax1 = plt.subplot(2, 2, 1)
            plt.plot(df['timestamp'], df['value'], linewidth=2, marker='o', markersize=4)
            plt.title('Monthly Values (24 Months)')
            plt.ylabel(f'{self.metric} ({self.unit})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plot_count += 1
        
        # Medium-term plot (hourly data)
        if 'hourly_1month' in self.data and not self.data['hourly_1month'].empty:
            df = self.data['hourly_1month']
            df['hour'] = df['timestamp'].dt.hour
            
            # Hourly pattern line plot
            ax2 = plt.subplot(2, 2, 2)
            hourly_avg = df.groupby('hour')['value'].mean()
            plt.plot(hourly_avg.index, hourly_avg.values, linewidth=2, marker='o', markersize=4)
            plt.title('Average Hourly Pattern (1 Month)')
            plt.xlabel('Hour of Day')
            plt.ylabel(f'Average {self.unit}')
            plt.grid(True, alpha=0.3)
            plot_count += 1
            
            # Hourly time series
            ax3 = plt.subplot(2, 2, 3)
            plt.plot(df['timestamp'], df['value'], linewidth=1, alpha=0.7)
            plt.title('Hourly Time Series (1 Month)')
            plt.ylabel(f'{self.metric} ({self.unit})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plot_count += 1
        
        # Short-term plot (5-minute data)
        if 'fivemin_1week' in self.data and not self.data['fivemin_1week'].empty:
            df = self.data['fivemin_1week']
            
            # 5-minute time series line plot
            ax4 = plt.subplot(2, 2, 4)
            plt.plot(df['timestamp'], df['value'], linewidth=0.8, alpha=0.8)
            plt.title('5-Minute Time Series (1 Week)')
            plt.ylabel(f'{self.metric} ({self.unit})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plot_count += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Line plot visualizations saved to {save_path}")
        
        plt.show()

def run_dual_analytics(power_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame], 
                      save_report: bool = True, save_plots: bool = True) -> DualMetricAnalytics:
    """
    Main function to run dual analytics for power and market value data
    
    Args:
        power_data: Dictionary containing power datasets
        market_data: Dictionary containing market value datasets
        save_report: Whether to save text report
        save_plots: Whether to save visualizations
        
    Returns:
        DualMetricAnalytics instance with results
    """
    # Create analytics directory
    analytics_dir = '../Analytics'
    os.makedirs(analytics_dir, exist_ok=True)
    
    # Initialize dual analytics
    dual_analytics = DualMetricAnalytics(power_data, market_data)
    
    # Generate and display report
    report = dual_analytics.generate_dual_report()
    print(report)
    
    if save_report:
        # Use fixed filename instead of timestamp
        report_filename = f"{analytics_dir}/dual_energy_report.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Dual report saved to {report_filename}")
    
    if save_plots:
        # Use fixed filename instead of timestamp
        plot_filename = f"{analytics_dir}/dual_energy_analysis.png"
        dual_analytics.create_dual_visualizations(plot_filename)
    
    return dual_analytics

def run_analytics(data: Dict[str, pd.DataFrame], save_report: bool = True, save_plots: bool = True) -> EnergyAnalytics:
    """
    Main function to run comprehensive analytics
    
    Args:
        data: Dictionary containing energy datasets
        save_report: Whether to save text report
        save_plots: Whether to save visualizations
        
    Returns:
        EnergyAnalytics instance with results
    """
    # Create analytics directory
    analytics_dir = '../Analytics'
    os.makedirs(analytics_dir, exist_ok=True)
    
    # Initialize analytics
    analytics = EnergyAnalytics(data)
    
    # Generate and display report
    report = analytics.generate_report()
    print(report)
    
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{analytics_dir}/energy_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Report saved to {report_filename}")
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{analytics_dir}/energy_analysis_{timestamp}.png"
        analytics.create_visualizations(plot_filename)
    
    return analytics

if __name__ == "__main__":
    print("🔬 Energy Analytics Module")
    print("Use this module with data from fetch.py")