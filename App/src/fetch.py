#!/usr/bin/env python3
"""
Data Fetcher for Energy Analytics
Fetches the three required datasets:
- Monthly data for past 24 months
- Hourly data for past month
- 5-minute data for past week
"""

from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import os
from typing import Dict, Optional

# Load environment variables BEFORE importing openelectricity
# Try multiple credential file locations for different environments
credential_paths = ['../credentials.txt', '../../App/credentials.txt', './credentials.txt']
for path in credential_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break
else:
    # If no credentials file found, assume environment variables are already set
    pass

from openelectricity import OEClient
from openelectricity.types import DataMetric

class EnergyDataFetcher:
    """Simplified data fetcher for energy analytics"""
    
    def __init__(self, network_code: str = 'NEM'):
        """Initialize the fetcher"""
        self.client = OEClient()
        self.network_code = network_code
        self.data_dir = '../Data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_data(self, metric: str = 'POWER', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required datasets
        
        Args:
            metric: Data metric to fetch (POWER, ENERGY, etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing the three datasets
        """
        
        datasets = {
            'monthly_24months': {'interval': '1M', 'months_back': 23, 'exclude_current_month': True},
            'hourly_1month': {'interval': '1h', 'days_back': 30},
            'fivemin_1week': {'interval': '5m', 'days_back': 7}
        }
        
        results = {}
        
        for dataset_key, config in datasets.items():
            print(f"🔄 Fetching {dataset_key}...")
            
            # Check cache first if requested
            if use_cache:
                cached_df = self._load_from_cache(dataset_key, metric)
                if cached_df is not None:
                    results[dataset_key] = cached_df
                    print(f"📁 Loaded {dataset_key} from cache ({len(cached_df)} records)")
                    continue
            
            # Fetch fresh data
            df = self._fetch_dataset(dataset_key, config, metric)
            if df is not None:
                results[dataset_key] = df
                self._save_to_cache(df, dataset_key, metric)
                print(f"✅ Fetched {dataset_key} ({len(df)} records)")
            else:
                print(f"❌ Failed to fetch {dataset_key}")
        
        return results
    
    def _fetch_dataset(self, dataset_key: str, config: dict, metric: str) -> Optional[pd.DataFrame]:
        """Fetch a single dataset from the API"""
        try:
            # Calculate date range
            end_date = datetime.now()
            
            # For monthly data, exclude current month and go back exactly N months
            if config.get('exclude_current_month', False) and 'months_back' in config:
                # Set end_date to first day of previous month
                current_date = datetime.now()
                if current_date.month == 1:
                    # If January, previous month is December of previous year
                    end_date = datetime(current_date.year - 1, 12, 1, 0, 0, 0)
                else:
                    # Get first day of previous month
                    prev_month = current_date.month - 1
                    year = current_date.year
                    end_date = datetime(year, prev_month, 1, 0, 0, 0)
                
                # Calculate start_date by going back exactly N months from previous month
                months_back = config['months_back']
                start_year = end_date.year
                start_month = end_date.month
                
                # Go back the specified number of months
                total_months = start_year * 12 + start_month - months_back
                start_year = total_months // 12
                start_month = total_months % 12
                if start_month == 0:
                    start_month = 12
                    start_year -= 1
                
                # Set to first day of that start month
                start_date = datetime(start_year, start_month, 1)
                
            else:
                # For non-monthly data, use days_back as before
                start_date = end_date - timedelta(days=config['days_back'])
            
            # Convert metric string to DataMetric enum
            metric_enum = getattr(DataMetric, metric.upper(), None)
            if not metric_enum:
                raise ValueError(f"Invalid metric: {metric}")
            
            # Fetch data from API
            response = self.client.get_network_data(
                network_code=self.network_code,
                metrics=[metric_enum],
                interval=config['interval'],
                date_start=start_date,
                date_end=end_date
            )
            
            if not response or not response.data:
                return None
            
            # Convert to DataFrame
            df_data = []
            for series in response.data:
                if series.results:
                    for result in series.results:
                        for point in result.data:
                            timestamp, value = point.root
                            df_data.append({
                                'timestamp': timestamp,
                                'value': value,
                                'metric': series.metric,
                                'unit': series.unit,
                                'interval': config['interval'],
                                'network': series.network_code
                            })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                return df
            
            return None
            
        except Exception as e:
            print(f"❌ Error fetching {dataset_key}: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, dataset_key: str, metric: str) -> None:
        """Save DataFrame to cache file with fixed filename"""
        try:
            # Use fixed filename instead of timestamp
            filename = f"{self.network_code}_{metric}_{dataset_key}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Add metadata
            df_save = df.copy()
            df_save['fetch_timestamp'] = datetime.now()
            df_save['dataset_type'] = dataset_key
            
            df_save.to_csv(filepath, index=False)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save cache for {dataset_key}: {e}")
    
    def _load_from_cache(self, dataset_key: str, metric: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache file with fixed filename"""
        try:
            # Use fixed filename pattern
            filename = f"{self.network_code}_{metric}_{dataset_key}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                return None
            
            # Check if file is fresh (less than 6 hours old)
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            if age_hours > 6:  # Cache expired
                return None
            
            # Load and return data
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load cache for {dataset_key}: {e}")
            return None

def fetch_energy_data(network: str = 'NEM', metric: str = 'POWER', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Main function to fetch energy data
    
    Args:
        network: Network code (NEM, WEM, AU)
        metric: Data metric (POWER, ENERGY, etc.)
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary with datasets: monthly_24months, hourly_1month, fivemin_1week
    """
    fetcher = EnergyDataFetcher(network)
    return fetcher.fetch_data(metric, use_cache)

def fetch_power_and_market_data(network: str = 'NEM', use_cache: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch both POWER and MARKET_VALUE data
    
    Args:
        network: Network code (NEM, WEM, AU)
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary with structure: {'power': {datasets}, 'market_value': {datasets}}
    """
    print("🔌 Fetching POWER data...")
    power_data = fetch_energy_data(network, 'POWER', use_cache)
    
    print("💰 Fetching MARKET_VALUE data...")
    market_data = fetch_energy_data(network, 'MARKET_VALUE', use_cache)
    
    return {
        'power': power_data,
        'market_value': market_data
    }

# For backward compatibility with existing code
def get_power_data(network_code: str = 'NEM', force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Get power data - maintains compatibility with existing code"""
    return fetch_energy_data(network_code, 'POWER', not force_refresh)

if __name__ == "__main__":
    # Test the fetcher
    print("🔌 Testing Energy Data Fetcher")
    print("=" * 40)
    
    data = fetch_energy_data('NEM', 'POWER', use_cache=True)
    
    if data:
        print(f"\n📊 Successfully fetched {len(data)} datasets:")
        for dataset_key, df in data.items():
            date_range = df['timestamp'].max() - df['timestamp'].min()
            print(f"  • {dataset_key}: {len(df)} records over {date_range.days} days")
    else:
        print("❌ No data retrieved")