#!/usr/bin/env python3
"""
Dual Energy Analytics App
Analyzes both Power and Market Value data with combined visualizations
"""

from fetch import fetch_power_and_market_data
from analytics import run_dual_analytics

def main():
    """Main function for dual analytics"""
    
    print("⚡ DUAL ENERGY ANALYTICS")
    print("=" * 40)
    print("🔌 NEM Power & Market Value Analysis")
    print("📊 Monthly (24mo) + Hourly (1mo) + 5min (1wk)")
    print("📈 Combined line plot visualizations")
    
    try:
        # Ask user if they want to proceed
        confirm = input("\n▶ Run dual analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("⏹ Cancelled.")
            return
        
        print("\n🔄 Fetching power and market value data...")
        
        # Fetch both power and market value data
        all_data = fetch_power_and_market_data('NEM', use_cache=True)
        power_data = all_data.get('power', {})
        market_data = all_data.get('market_value', {})
        
        if not power_data and not market_data:
            print("❌ No data available. Check your internet connection and API credentials.")
            return
        
        print(f"✅ Got Power: {len(power_data)} datasets, Market Value: {len(market_data)} datasets")
        
        print("\n🔬 Running dual analytics...")
        
        # Run comprehensive dual analytics
        dual_analytics = run_dual_analytics(
            power_data=power_data,
            market_data=market_data,
            save_report=True, 
            save_plots=True
        )
        
        print(f"\n✅ DUAL ANALYSIS COMPLETE!")
        print(f"� Check Analytics/ folder for:")
        print(f"   • Detailed dual metric report")
        print(f"   • Combined power vs market value charts")
        print(f"� Generated combined line plot visualizations")
        
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have:")
        print("   • Internet connection")
        print("   • Valid API key in credentials.txt")
        print("   • Required packages installed")

if __name__ == "__main__":
    main()