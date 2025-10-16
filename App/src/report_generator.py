#!/usr/bin/env python3
"""
Report Generator Module
Generate dynamic HTML reports with current data statistics
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional

# Import utilities and configuration
from utils import DataUtils, FileUtils, FormatUtils, TrendAnalyzer
from config import Config, ErrorMessages, ConsoleMessages

class EnergyReportGenerator:
    """Generate comprehensive energy data reports"""
    
    def __init__(self, power_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]):
        """Initialize with power and market data"""
        self.power_data = power_data
        self.market_data = market_data
        self.network = DataUtils.get_network_name(power_data)
        self.docs_dir = FileUtils.get_docs_path()
        FileUtils.ensure_directory(self.docs_dir)
    
    def _calculate_all_stats(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate statistics for all datasets using utilities"""
        stats = {}
        for dataset_key, df in data_dict.items():
            if DataUtils.validate_dataframe(df):
                stats[dataset_key] = DataUtils.calculate_basic_stats(df)
            else:
                stats[dataset_key] = {'error': 'Invalid or empty dataset'}
        return stats
    
    def _analyze_dataset_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends using TrendAnalyzer utilities"""
        if not DataUtils.validate_dataframe(df, ['value']):
            return {'error': 'No data for trend analysis'}
        
        return TrendAnalyzer.analyze_dataset_trends(df)
    
    def _build_html_report(self, report_data: Dict) -> str:
        """Build complete HTML report from data"""
        html_parts = [
            self._get_html_header(report_data['timestamp']),
            self._build_data_section('Power', report_data['power_stats'], report_data['power_trends'], 'MW'),
            self._build_data_section('Market Value', report_data['market_stats'], report_data['market_trends'], 'AUD'),
            self._get_html_footer()
        ]
        return ''.join(html_parts)
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive HTML report with current data using clean template format"""
        
        # Calculate statistics for all datasets using utilities
        power_stats = self._calculate_all_stats(self.power_data)
        market_stats = self._calculate_all_stats(self.market_data)
        
        # Analyze trends using TrendAnalyzer
        power_trends = self._analyze_dataset_trends(self.power_data.get('monthly_24months', pd.DataFrame()))
        market_trends = self._analyze_dataset_trends(self.market_data.get('monthly_24months', pd.DataFrame()))
        
        # Generate report data
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'network': self.network,
            'power_stats': power_stats,
            'market_stats': market_stats,
            'power_trends': power_trends,
            'market_trends': market_trends,
            'totals': {
                'power_records': sum(len(df) for df in self.power_data.values()),
                'market_records': sum(len(df) for df in self.market_data.values())
            }
        }
        
        return self._build_html_report(report_data)
    
    def _get_html_header(self, timestamp: str) -> str:
        """Generate HTML header with CSS"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{Config.REPORT_CONFIG['title']}</title>
    {self._get_css_styles()}
</head>
<body>
    <div class="container">
        <div class="technical-report-link">
            <a href="./index.html">📊 Back to Dashboard</a>
        </div>
        <h1>{Config.REPORT_CONFIG['title']}</h1>
        <div class="timestamp">
            📊 Generated: {timestamp}<br>
            {Config.REPORT_CONFIG['update_message']}
        </div>"""
    
    def _get_html_footer(self) -> str:
        """Generate HTML footer"""
        return """
        <div class="technical-report-link">
            <a href="./index.html">📊 Back to Dashboard</a>
        </div>
    </div>
</body>
</html>"""
    
    def _build_data_section(self, section_title: str, stats: Dict, trends: Dict, unit_suffix: str) -> str:
        """Build a complete data section"""
        icon = "🔌" if "Power" in section_title else "💰"
        metric_name = "power" if "Power" in section_title else "market_value"
        unit_display = "MW" if unit_suffix == "MW" else "AUD ($)"
        
        section = f"""
        <h2>{icon} {section_title} Data Analysis</h2>
        <div class="metric-info">
            <strong>Network:</strong> {self.network} &nbsp;&nbsp;
            <strong>Metric:</strong> {metric_name} &nbsp;&nbsp;
            <strong>Unit:</strong> {unit_display}
        </div>"""
        
        # Add dataset sections
        for dataset_key, dataset_stats in stats.items():
            if 'error' not in dataset_stats:
                section += self._build_dataset_section(dataset_key, dataset_stats, unit_suffix)
        
        # Add insights section
        if 'error' not in trends:
            section += self._build_insights_section(section_title, trends)
        
        return section
    
    def _build_insights_section(self, section_title: str, trends: Dict) -> str:
        """Build insights section with trend analysis"""
        return f"""
        <div class="insights-section">
            <div class="insights-title">💡 {section_title} Insights</div>
            <div class="insight-item">Trend Direction: {trends.get('trend_direction', 'Unknown').title()}</div>
            <div class="insight-item">Peak Month: {trends.get('peak_month', 'Unknown')}</div>
            <div class="insight-item">Low Month: {trends.get('low_month', 'Unknown')}</div>
            <div class="insight-item">Volatility Index: {trends.get('volatility', 0):.3f}</div>
        </div>"""
    
    def _get_css_styles(self) -> str:
        """Return CSS styles using configuration"""
        colors = Config.get_ui_colors()
        fonts = Config.UI_CONFIG['fonts']
        spacing = Config.UI_CONFIG['spacing']
        
        return f"""<style>
        body {{
            font-family: {fonts['main']};
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            background: white;
            padding: {spacing['container_padding']};
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: {colors['primary']};
            text-align: center;
            margin-bottom: 10px;
            font-size: {fonts['title_size']};
            font-weight: 700;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-style: italic;
            background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {colors['primary']};
        }}
        h2 {{
            color: {colors['secondary']};
            border-bottom: 3px solid {colors['secondary']};
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 1.8em;
        }}
        .metric-info {{
            background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
            padding: 20px;
            border-radius: 10px;
            margin: {spacing['item_gap']} 0;
            border-left: 5px solid {colors['primary']};
            font-size: 1.1em;
        }}
        .dataset-section {{
            margin: {spacing['section_margin']} 0;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            border-left: 4px solid {colors['success']};
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .dataset-title {{
            font-size: 1.4em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: {spacing['item_gap']};
            margin: {spacing['item_gap']} 0;
        }}
        .stat-item {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        .stat-item:hover {{ transform: translateY(-2px); }}
        .stat-label {{
            font-weight: bold;
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 1.3em;
            color: {colors['primary']};
            margin-top: 8px;
            font-weight: 600;
        }}
        .insights-section {{
            background: linear-gradient(135deg, {colors['warning_bg']} 0%, #ffeaa7 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 25px 0;
            border-left: 5px solid {colors['warning']};
        }}
        .insights-title {{
            font-weight: bold;
            color: {colors['warning_text']};
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .insight-item {{
            margin: 10px 0;
            padding-left: 25px;
            position: relative;
            font-size: 1.05em;
        }}
        .insight-item:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: {colors['warning']};
            font-weight: bold;
            font-size: 1.2em;
        }}
        .technical-report-link {{ 
            text-align: center; 
            margin: 25px 0; 
        }}
        .technical-report-link a {{ 
            background: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
            color: white; 
            text-decoration: none; 
            font-weight: bold; 
            font-size: {fonts['button_size']}; 
            padding: 15px 30px;
            border-radius: 25px;
            transition: transform 0.3s ease;
            display: inline-block;
        }}
        .technical-report-link a:hover {{ transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.2); }}
    </style>"""
    
    def _build_dataset_section(self, dataset_key: str, stats: Dict, unit_suffix: str) -> str:
        """Build a dataset section with formatted statistics"""
        dataset_display = FormatUtils.get_dataset_display_name(dataset_key)
        
        section = f"""
        <div class="dataset-section">
            <div class="dataset-title">{dataset_display}</div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Records</div>
                    <div class="stat-value">{stats['count']:,}</div>
                </div>"""
        
        # Add date range if available
        if stats.get('date_range'):
            start_date = stats['date_range']['start']
            end_date = stats['date_range']['end']
            date_range = FormatUtils.format_date_range(start_date, end_date)
            section += f"""
                <div class="stat-item">
                    <div class="stat-label">Period</div>
                    <div class="stat-value">{date_range}</div>
                </div>"""
        
        # Format values based on unit type
        if unit_suffix == 'AUD':
            min_val = FormatUtils.format_currency(stats['min'])
            max_val = FormatUtils.format_currency(stats['max'])
            avg_val = FormatUtils.format_currency(stats['mean'])
            std_val = FormatUtils.format_currency(stats['std'])
        else:  # MW
            min_val = FormatUtils.format_power(stats['min'])
            max_val = FormatUtils.format_power(stats['max'])
            avg_val = FormatUtils.format_power(stats['mean'])
            std_val = FormatUtils.format_power(stats['std'])
        
        section += f"""
                <div class="stat-item">
                    <div class="stat-label">Minimum</div>
                    <div class="stat-value">{min_val}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Maximum</div>
                    <div class="stat-value">{max_val}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Average</div>
                    <div class="stat-value">{avg_val}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Std Deviation</div>
                    <div class="stat-value">{std_val}</div>
                </div>
            </div>
        </div>"""
        
        return section
    
    def save_report(self) -> str:
        """Generate and save the comprehensive report"""
        try:
            report_html = self.generate_comprehensive_report()
            report_path = os.path.join(self.docs_dir, 'dual_energy_report.html')
            
            if FileUtils.save_json({'html': report_html}, report_path.replace('.html', '_temp.json')):
                # Save as HTML file
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_html)
                
                print(f"📄 Dynamic report saved to {report_path}")
                return report_path
            else:
                raise Exception("Failed to save report data")
            
        except Exception as e:
            print(ErrorMessages.format_error('FILE_SAVE_ERROR', 
                                           filepath=report_path, error=str(e)))
            return ""
    
def generate_energy_report(power_data: Dict[str, pd.DataFrame], 
                          market_data: Dict[str, pd.DataFrame]) -> str:
    """
    Main function to generate energy data report
    
    Args:
        power_data: Dictionary containing power datasets
        market_data: Dictionary containing market datasets
        
    Returns:
        Path to generated report file
    """
    generator = EnergyReportGenerator(power_data, market_data)
    return generator.save_report()

if __name__ == "__main__":
    print("📊 Energy Report Generator Module")
    print("Use this module with data from fetch.py")