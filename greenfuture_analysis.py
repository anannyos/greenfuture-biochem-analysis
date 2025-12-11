# ==========================================
# Product Master Analysis (SKU & Brand Focused)
# Author: Shoumika Anannyo
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import squarify

# --- 1. Load Product Master Dataset ---
path = "/Users/shoumikaanannyo/Downloads/ProductMasterforAnalysis1.xlsx"
product_df = pd.read_excel(path)  # Remove sheet_name if it's the only sheet

# --- 2. Clean Columns ---
product_df.columns = product_df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
print("\nProduct Master Columns detected:", product_df.columns.tolist())
print("First few rows:\n", product_df.head())

# --- 3. Data Quality Checks ---
print("\n===== DATA QUALITY CHECK =====")
print("Total SKUs:", len(product_df))
print("Missing Values:\n", product_df.isnull().sum())
print("\nDuplicate SKUs:", product_df['SKU_Code'].duplicated().sum())

# --- 4. Descriptive Statistics ---
print("\n===== PRODUCT MASTER OVERVIEW =====")
print("Total Unique Brands:", product_df['Brand'].nunique())
print("Total Categories:", product_df['Category'].nunique())

print("\nProducts by Category:\n", product_df['Category'].value_counts())
print("\nProducts by Brand:\n", product_df['Brand'].value_counts())

# --- 5. Setup Output Path ---
save_dir = "/Users/shoumikaanannyo/Desktop/Product_Master_Visuals"
os.makedirs(save_dir, exist_ok=True)

sns.set(style="whitegrid", palette="pastel")

# --- 6. VISUALIZATIONS ---

## 6.1 Products by Category (Bar Chart)
plt.figure(figsize=(10, 6))
category_counts = product_df['Category'].value_counts()
sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis")
plt.title("Product Distribution by Category")
plt.xlabel("Number of Products")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(f"{save_dir}/products_by_category.png", dpi=300, bbox_inches='tight')
plt.close()

## 6.2 Products by Brand (Top 15)
plt.figure(figsize=(12, 8))
brand_counts = product_df['Brand'].value_counts().head(15)
sns.barplot(x=brand_counts.values, y=brand_counts.index, palette="magma")
plt.title("Top 15 Brands by Product Count")
plt.xlabel("Number of Products")
plt.ylabel("Brand")
plt.tight_layout()
plt.savefig(f"{save_dir}/top_brands_by_count.png", dpi=300, bbox_inches='tight')
plt.close()

## 6.3 Category-Brand Heatmap
plt.figure(figsize=(12, 8))
cross_tab = pd.crosstab(product_df['Brand'], product_df['Category'])
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Products'})
plt.title("Brand-Category Distribution Heatmap")
plt.xlabel("Category")
plt.ylabel("Brand")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/brand_category_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

## 6.4 Brand Diversification (Pie Chart)
plt.figure(figsize=(10, 8))
brand_diversity = product_df.groupby('Brand')['Category'].nunique().sort_values(ascending=False)
plt.pie(brand_diversity.values, labels=brand_diversity.index, autopct='%1.1f%%', startangle=90)
plt.title("Brand Diversification Across Categories")
plt.tight_layout()
plt.savefig(f"{save_dir}/brand_diversification_pie.png", dpi=300, bbox_inches='tight')
plt.close()

## 6.5 Category Composition (Treemap)
plt.figure(figsize=(12, 8))
category_sizes = product_df['Category'].value_counts()
squarify.plot(sizes=category_sizes.values, label=category_sizes.index, alpha=0.8, 
              color=sns.color_palette("Spectral", len(category_sizes)))
plt.title("Product Category Distribution (Treemap)")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{save_dir}/category_treemap.png", dpi=300, bbox_inches='tight')
plt.close()

## 6.6 SKU Pattern Analysis
plt.figure(figsize=(10, 6))
# Extract prefix from SKU codes (first 3 parts before numbers)
product_df['SKU_Prefix'] = product_df['SKU_Code'].str.extract(r'([A-Z]+)-?')[0]
sku_prefix_counts = product_df['SKU_Prefix'].value_counts().head(10)
sns.barplot(x=sku_prefix_counts.values, y=sku_prefix_counts.index, palette="coolwarm")
plt.title("Top 10 SKU Prefix Patterns")
plt.xlabel("Count")
plt.ylabel("SKU Prefix")
plt.tight_layout()
plt.savefig(f"{save_dir}/sku_prefix_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 7. Advanced Analytics ---
print("\n===== ADVANCED ANALYTICS =====")

# Brand concentration analysis
brand_concentration = product_df['Brand'].value_counts(normalize=True) * 100
print(f"\nTop 5 Brands Concentration: {brand_concentration.head(5).to_dict()}")

# Category spread analysis
category_spread = product_df.groupby('Category')['Brand'].nunique().sort_values(ascending=False)
print(f"\nBrand Diversity per Category:\n{category_spread}")

# SKU naming consistency
print(f"\nSKU Naming Patterns Found: {product_df['SKU_Prefix'].nunique()} unique prefixes")

# --- 8. Summary Report ---
print("\n" + "="*50)
print("PRODUCT MASTER SUMMARY REPORT")
print("="*50)
print(f"Total Products: {len(product_df)}")
print(f"Unique Brands: {product_df['Brand'].nunique()}")
print(f"Categories: {product_df['Category'].nunique()}")
print(f"Most Popular Category: {product_df['Category'].value_counts().index[0]} ({product_df['Category'].value_counts().iloc[0]} products)")
print(f"Most Prolific Brand: {product_df['Brand'].value_counts().index[0]} ({product_df['Brand'].value_counts().iloc[0]} products)")
print(f"Data Quality: {product_df['SKU_Code'].duplicated().sum()} duplicate SKUs, {product_df['SKU_Code'].isnull().sum()} missing SKUs")
print(f"Visualizations saved to: {save_dir}")
print("="*50)

# --- 9. Export Summary Statistics ---
summary_stats = {
    'total_products': len(product_df),
    'unique_brands': product_df['Brand'].nunique(),
    'categories': product_df['Category'].nunique(),
    'top_category': product_df['Category'].value_counts().index[0],
    'top_brand': product_df['Brand'].value_counts().index[0],
    'duplicate_skus': product_df['SKU_Code'].duplicated().sum(),
    'missing_skus': product_df['SKU_Code'].isnull().sum()
}

# Save summary to CSV
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(f"{save_dir}/product_master_summary.csv", index=False)

print("\nProduct Master analysis completed!")
print(f"{len(os.listdir(save_dir))} visualizations saved")
print(f"Summary statistics exported to CSV")


# ==========================================
# Product Master Dataset Analysis
# Author: Shoumika Anannyo
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. Load Data ---
path = "/Users/shoumikaanannyo/Downloads/ProductMasterforAnalysis1.xlsx"
df = pd.read_excel(path)
df.columns = df.columns.str.strip().str.replace(" ", "_")
print("Columns:", df.columns.tolist())

# --- 2. Basic Overview ---
print("\n===== PRODUCT MASTER OVERVIEW =====")
print("Total Records:", len(df))
print("Unique SKUs:", df['SKU_Code'].nunique())
print("\nSKUs by Brand:\n", df['Brand'].value_counts())
print("\nSKUs by Category:\n", df['Category'].value_counts())

# --- 3. Brand-Category Crosstab ---
crosstab = pd.crosstab(df['Brand'], df['Category'])
print("\n===== BRAND × CATEGORY MATRIX =====")
print(crosstab)

# --- 4. Visualization Setup ---
save_dir = "/Users/shoumikaanannyo/Desktop/ProductMaster_Visuals"
os.makedirs(save_dir, exist_ok=True)
sns.set(style="whitegrid", palette="muted")

# --- 5. Visualizations ---

## 5.1 SKUs by Brand
plt.figure(figsize=(10,6))
sns.countplot(data=df, y='Brand', order=df['Brand'].value_counts().index)
plt.title("SKUs by Brand")
plt.xlabel("Count of SKUs")
plt.ylabel("Brand")
plt.tight_layout()
plt.savefig(f"{save_dir}/skus_by_brand.png")
plt.close()

## 5.2 SKUs by Category
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Category', order=df['Category'].value_counts().index)
plt.title("SKUs by Category")
plt.xlabel("Category")
plt.ylabel("Count of SKUs")
plt.tight_layout()
plt.savefig(f"{save_dir}/skus_by_category.png")
plt.close()

## 5.3 Brand vs Category Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(crosstab, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Brand × Category Relationship")
plt.tight_layout()
plt.savefig(f"{save_dir}/brand_category_heatmap.png")
plt.close()

## 5.4 Top 10 Brands (if many)
if df['Brand'].nunique() > 10:
    top_brands = df['Brand'].value_counts().nlargest(10).index
    plt.figure(figsize=(8,5))
    sns.countplot(data=df[df['Brand'].isin(top_brands)], y='Brand', order=top_brands)
    plt.title("Top 10 Brands by SKU Count")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top10_brands.png")
    plt.close()

print(f"\nVisualizations saved to: {save_dir}")
print("Summary stats printed above.\n")


# ==========================================
# Manufacturing Production Analysis
# Author: Shoumika Anannyo
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- 1. Load Data ---
path = "/Users/shoumikaanannyo/Downloads/ManufacturingProductionAnalysis.xlsx"
df = pd.read_excel(path)  # No sheet_name needed if it's the only sheet

# --- 2. Clean Columns & Data Types ---
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_").str.replace("(", "").str.replace(")", "")
print("Columns:", df.columns.tolist())
print("First few rows:\n", df.head())

# Convert date and calculate derived metrics
df['Posting_Date'] = pd.to_datetime(df['Posting_Date'], errors='coerce')
df['Month'] = df['Posting_Date'].dt.month_name()
df['Quarter'] = df['Posting_Date'].dt.quarter
df['Cost_Variance_%'] = ((df['Actual_Cost_per_MT_$'] - df['Std_Cost_per_MT_$']) / df['Std_Cost_per_MT_$']) * 100
df['Quantity_Variance_%'] = ((df['Actual_Quantity_MT'] - df['Planned_Quantity_MT']) / df['Planned_Quantity_MT']) * 100

# --- 3. Descriptive Statistics ---
print("\n" + "="*60)
print("MANUFACTURING PRODUCTION - DESCRIPTIVE STATISTICS")
print("="*60)

print(f"\nBASIC OVERVIEW:")
print(f"Total Production Orders: {len(df)}")
print(f"Date Range: {df['Posting_Date'].min().strftime('%Y-%m-%d')} to {df['Posting_Date'].max().strftime('%Y-%m-%d')}")
print(f"Plants: {df['Plant_Code'].nunique()} ({', '.join(df['Plant_Code'].unique())})")
print(f"Unique Materials: {df['Material_Code'].nunique()}")

print(f"\nPRODUCTION BY PLANT:")
plant_summary = df.groupby('Plant_Code').agg({
    'ProdOrder_ID': 'count',
    'Planned_Quantity_MT': 'sum',
    'Actual_Quantity_MT': 'sum',
    'Yield_%': 'mean'
}).round(2)
print(plant_summary)

print(f"\nCOST ANALYSIS:")
cost_stats = df[['Std_Cost_per_MT_$', 'Actual_Cost_per_MT_$', 'Cost_Variance_%']].describe()
print(cost_stats)

print(f"\nYIELD PERFORMANCE:")
yield_stats = df['Yield_%'].describe()
print(yield_stats)
print(f"\nOrders with Yield > 100%: {(df['Yield_%'] > 100).sum()}")  # Overproduction
print(f"Orders with Yield < 90%: {(df['Yield_%'] < 90).sum()}")     # Underperformance

print(f"\nQUANTITY VARIANCE:")
quantity_stats = df['Quantity_Variance_%'].describe()
print(quantity_stats)

# --- 4. Setup Output Directory ---
save_dir = "/Users/shoumikaanannyo/Desktop/Manufacturing_Visuals"
os.makedirs(save_dir, exist_ok=True)
sns.set(style="whitegrid", palette="viridis")

# --- 5. VISUALIZATIONS ---

## 5.1 Yield Distribution by Plant
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Plant_Code', y='Yield_%')
plt.title('Yield Distribution by Plant')
plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Target Yield (100%)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/yield_by_plant.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.2 Cost Variance Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Std_Cost_per_MT_$', y='Actual_Cost_per_MT_$', hue='Plant_Code', s=100, alpha=0.7)
plt.plot([df['Std_Cost_per_MT_$'].min(), df['Std_Cost_per_MT_$'].max()], 
         [df['Std_Cost_per_MT_$'].min(), df['Std_Cost_per_MT_$'].max()], 'r--', alpha=0.5, label='Ideal Line')
plt.title('Standard vs Actual Cost Analysis')
plt.xlabel('Standard Cost per MT ($)')
plt.ylabel('Actual Cost per MT ($)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/cost_variance_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.3 Monthly Production Volume
plt.figure(figsize=(12, 6))
monthly_production = df.groupby('Month')['Actual_Quantity_MT'].sum()
# Reorder months chronologically if possible
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_production = monthly_production.reindex([month for month in month_order if month in monthly_production.index])
monthly_production.plot(kind='bar', color='skyblue')
plt.title('Monthly Production Volume (Actual Quantity)')
plt.xlabel('Month')
plt.ylabel('Total Production (MT)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/monthly_production.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.4 Plant Efficiency Heatmap
plt.figure(figsize=(10, 8))
efficiency_matrix = df.pivot_table(
    values='Yield_%', 
    index='Plant_Code', 
    columns=pd.cut(df['Posting_Date'].dt.month, bins=3, labels=['Q1', 'Q2', 'Q3']), 
    aggfunc='mean'
)
sns.heatmap(efficiency_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=100)
plt.title('Plant Efficiency Heatmap (Yield % by Quarter)')
plt.tight_layout()
plt.savefig(f"{save_dir}/plant_efficiency_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.5 Cost Variance Distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Cost_Variance_%'], kde=True, bins=20)
plt.title('Cost Variance Distribution')
plt.xlabel('Cost Variance (%)')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='Plant_Code', x='Cost_Variance_%')
plt.title('Cost Variance by Plant')
plt.tight_layout()
plt.savefig(f"{save_dir}/cost_variance_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.6 Top Materials by Production Volume
plt.figure(figsize=(12, 6))
top_materials = df.groupby('Material_Name')['Actual_Quantity_MT'].sum().nlargest(10)
sns.barplot(x=top_materials.values, y=top_materials.index)
plt.title('Top 10 Materials by Production Volume')
plt.xlabel('Total Production (MT)')
plt.tight_layout()
plt.savefig(f"{save_dir}/top_materials_production.png", dpi=300, bbox_inches='tight')
plt.close()

## 5.7 Yield vs Cost Correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Yield_%', y='Actual_Cost_per_MT_$', hue='Plant_Code', s=80)
plt.title('Yield vs Actual Cost Relationship')
plt.xlabel('Yield (%)')
plt.ylabel('Actual Cost per MT ($)')
plt.tight_layout()
plt.savefig(f"{save_dir}/yield_vs_cost.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 6. Advanced Analytics ---
print("\n" + "="*60)
print("ADVANCED PERFORMANCE ANALYTICS")
print("="*60)

# Plant ranking by efficiency
plant_ranking = df.groupby('Plant_Code').agg({
    'Yield_%': 'mean',
    'Cost_Variance_%': 'mean',
    'Actual_Quantity_MT': 'sum'
}).round(2)
plant_ranking['Efficiency_Score'] = (plant_ranking['Yield_%'] / 100) - (plant_ranking['Cost_Variance_%'] / 100).abs()
plant_ranking = plant_ranking.sort_values('Efficiency_Score', ascending=False)
print(f"\nPLANT EFFICIENCY RANKING:")
print(plant_ranking)

# Material performance analysis
material_performance = df.groupby('Material_Name').agg({
    'Yield_%': 'mean',
    'Cost_Variance_%': 'mean',
    'ProdOrder_ID': 'count'
}).round(2)
material_performance = material_performance.sort_values('Yield_%', ascending=False)
print(f"\nTOP PERFORMING MATERIALS:")
print(material_performance.head(10))

# --- 7. Export Summary ---
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Analysis completed for {len(df)} production orders")
print(f"{df['Plant_Code'].nunique()} plants analyzed")
print(f"{df['Material_Code'].nunique()} materials tracked")
print(f"Overall Average Yield: {df['Yield_%'].mean():.1f}%")
print(f"Overall Cost Variance: {df['Cost_Variance_%'].mean():.1f}%")
print(f"Visualizations saved to: {save_dir}")

# Export key metrics to CSV
summary_df = pd.DataFrame({
    'Metric': ['Total_Orders', 'Total_Plants', 'Total_Materials', 'Avg_Yield', 'Avg_Cost_Variance'],
    'Value': [len(df), df['Plant_Code'].nunique(), df['Material_Code'].nunique(), 
              df['Yield_%'].mean(), df['Cost_Variance_%'].mean()]
})
summary_df.to_csv(f"{save_dir}/manufacturing_summary.csv", index=False)

print(f"\nRECOMMENDATIONS:")
best_plant = plant_ranking.index[0]
worst_plant = plant_ranking.index[-1]
print(f"Best performing plant: {best_plant} (Efficiency Score: {plant_ranking.loc[best_plant, 'Efficiency_Score']:.3f})")
print(f"Focus improvement on: {worst_plant} (Efficiency Score: {plant_ranking.loc[worst_plant, 'Efficiency_Score']:.3f})")
print(f"{len(df[df['Yield_%'] < 90])} orders need yield improvement (<90%)")
print(f"{len(df[df['Cost_Variance_%'] > 5])} orders have significant cost overruns (>5%)")


# ==========================================
# GreenFuture BioChem: Enhanced Merge with Proper Linking
# Author: Shoumika Anannyo
# ==========================================

import pandas as pd
import os

# ---------- FILE PATHS ----------
path_rd = "/Users/shoumikaanannyo/Downloads/RDforAnalyses.xlsx"
path_product = "/Users/shoumikaanannyo/Downloads/ProductMasterforAnalysis1.xlsx"
path_mfg = "/Users/shoumikaanannyo/Downloads/ManufacturingProductionAnalysis.xlsx"
path_proc = "/Users/shoumikaanannyo/Downloads/SupplyChainAnalysis.xlsx"
path_sales = "/Users/shoumikaanannyo/Downloads/SalesPipelineAnalysis.xlsx"

save_dir = "/Users/shoumikaanannyo/Desktop/GreenFuture_Cleaned/"
os.makedirs(save_dir, exist_ok=True)

# ---------- Helper Function ----------
def clean_cols(df):
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("%", "Percent")
        .str.replace("$", "USD")
    )
    return df

print("LOADING AND CLEANING DATASETS...")

# ==========================================
# 1 R&D PIPELINE
# ==========================================
rd = pd.read_excel(path_rd)
rd = clean_cols(rd)
print("R&D Columns:", rd.columns.tolist())
rd["Est_Annual_Revenue_M"] = pd.to_numeric(rd["Est_Annual_Revenue_USDM"], errors="coerce")
rd["Est_Launch_Date"] = pd.to_datetime(rd["Est_Launch_Date"], errors="coerce")

# ==========================================
# 2 PRODUCT MASTER
# ==========================================
product = pd.read_excel(path_product)
product = clean_cols(product)
print("Product Columns:", product.columns.tolist())
product.drop_duplicates(subset=["SKU_Code"], inplace=True)

# ==========================================
# 3 MANUFACTURING PRODUCTION
# ==========================================
mfg = pd.read_excel(path_mfg)
mfg = clean_cols(mfg)
print("Manufacturing Columns:", mfg.columns.tolist())
for c in ["Planned_Quantity_MT", "Actual_Quantity_MT", "Yield_Percent",
          "Std_Cost_per_MT_USD", "Actual_Cost_per_MT_USD"]:
    mfg[c] = pd.to_numeric(mfg[c], errors="coerce")
mfg["Posting_Date"] = pd.to_datetime(mfg["Posting_Date"], errors="coerce")

# ==========================================
# 4 SUPPLY CHAIN PROCUREMENT
# ==========================================
proc = pd.read_excel(path_proc)
proc = clean_cols(proc)
print("Procurement Columns:", proc.columns.tolist())
for c in ["Qty_MT", "Unit_Cost_USD", "CO2_Emissions_kg_MT"]:
    proc[c] = pd.to_numeric(proc[c], errors="coerce")
proc["Delivery_Date"] = pd.to_datetime(proc["Delivery_Date"], errors="coerce")

# ==========================================
# 5 SALES PIPELINE
# ==========================================
sales = pd.read_excel(path_sales)
sales = clean_cols(sales)
print("Sales Columns:", sales.columns.tolist())
for c in ["Est_Value_USDM", "Probability_Percent"]:
    sales[c] = pd.to_numeric(sales[c], errors="coerce")
sales["Close_Date"] = pd.to_datetime(sales["Close_Date"], errors="coerce")

print("\nSTARTING ENHANCED MERGE STRATEGY...")

# ==========================================
# STRATEGY: Create linking through Product_Interest -> Material_Name -> SKU_Code
# ==========================================

# Step 1: Merge R&D with Sales (this worked)
print("1. Merging R&D with Sales...")
merged_data = rd.merge(sales, how="left", on="SalesOpp_ID", suffixes=('_RD', '_Sales'))
print(f"   R&D+Sales: {merged_data.shape}")

# Step 2: Link Product_Interest to Manufacturing Material_Name
print("2. Linking Product_Interest to Manufacturing data...")
# Create a mapping from Material_Name to Material_Code from manufacturing data
material_mapping = mfg[['Material_Code', 'Material_Name']].drop_duplicates()

# Merge using Product_Interest (from Sales) -> Material_Name (from Manufacturing)
merged_data = merged_data.merge(material_mapping, how="left", 
                                left_on="Product_Interest", 
                                right_on="Material_Name",
                                suffixes=('', '_Mfg'))

print(f"   Added Material_Code link: {merged_data.shape}")

# Step 3: Now merge with Product Master using Material_Code -> SKU_Code
print("3. Merging with Product Master...")
# Assuming Material_Code and SKU_Code are the same or can be mapped
# If they're different, we'll use the mapping we have
merged_data = merged_data.merge(product, how="left", 
                                left_on="Material_Code", 
                                right_on="SKU_Code",
                                suffixes=('', '_Product'))

print(f"   Added Product data: {merged_data.shape}")

# Step 4: Merge with Manufacturing using Material_Code
print("4. Merging with Manufacturing data...")
# Aggregate manufacturing data by Material_Code to avoid duplication
mfg_agg = mfg.groupby('Material_Code').agg({
    'Planned_Quantity_MT': 'sum',
    'Actual_Quantity_MT': 'sum',
    'Yield_Percent': 'mean',
    'Std_Cost_per_MT_USD': 'mean',
    'Actual_Cost_per_MT_USD': 'mean',
    'Plant_Code': lambda x: ', '.join(x.unique()),
    'ProdOrder_ID': 'count'
}).reset_index()
mfg_agg.rename(columns={'ProdOrder_ID': 'Manufacturing_Orders_Count'}, inplace=True)

merged_data = merged_data.merge(mfg_agg, how="left", on="Material_Code", suffixes=('', '_Mfg'))
print(f"   Added Manufacturing data: {merged_data.shape}")

# Step 5: Merge with Procurement using Material_Code
print("5. Merging with Procurement data...")
# Aggregate procurement data by Material_Code
proc_agg = proc.groupby('Material_Code').agg({
    'Qty_MT': 'sum',
    'Unit_Cost_USD': 'mean',
    'CO2_Emissions_kg_MT': 'mean',
    'Supplier_Name': lambda x: ', '.join(x.unique()),
    'PO_Number': 'count',
    'On_Time_Y_N': lambda x: (x == 'Y').mean() * 100  # On-time delivery rate
}).reset_index()
proc_agg.rename(columns={
    'PO_Number': 'Procurement_Orders_Count',
    'On_Time_Y_N': 'On_Time_Delivery_Percent'
}, inplace=True)

merged_data = merged_data.merge(proc_agg, how="left", on="Material_Code", suffixes=('', '_Proc'))
print(f"   Added Procurement data: {merged_data.shape}")

# ==========================================
# 6 FINAL CLEANUP AND EXPORT
# ==========================================
print("\nFINALIZING MERGED DATASET...")

# Remove duplicate columns
cols_to_keep = []
for col in merged_data.columns:
    if not col.endswith(('_Mfg', '_Product', '_Proc')) or col in ['Material_Code', 'SKU_Code']:
        cols_to_keep.append(col)

final_data = merged_data[cols_to_keep]

# Rename key columns for clarity
column_rename = {
    'Stage_RD': 'R&D_Stage',
    'Stage_Sales': 'Sales_Stage',
    'Project_ID_RD': 'Project_ID',
    'Project_ID_Sales': 'Linked_Project_ID'
}
final_data.rename(columns={k: v for k, v in column_rename.items() if k in final_data.columns}, inplace=True)

print(f"Final dataset shape: {final_data.shape}")
print(f"Total columns: {len(final_data.columns)}")

# ==========================================
# 7 EXPORT AND SUMMARY
# ==========================================
output_path = os.path.join(save_dir, "GreenFuture_Enhanced_Master.xlsx")
final_data.to_excel(output_path, index=False)

print(f"\nENHANCED MERGED DATASET SAVED AT:")
print(f"   {output_path}")

print(f"\nKEY METRICS IN FINAL DATASET:")
print(f"   R&D Projects: {len(rd)}")
print(f"   Sales Opportunities: {len(sales)}")
print(f"   Manufacturing Records: {len(mfg)}")
print(f"   Procurement Records: {len(proc)}")
print(f"   Final Linked Records: {len(final_data)}")

print(f"\nSUCCESSFULLY LINKED:")
print(f"   R&D ↔ Sales: {len(merged_data)} records")
print(f"   Product Master: {final_data['SKU_Code'].notna().sum()} records with product info")
print(f"   Manufacturing: {final_data['Manufacturing_Orders_Count'].notna().sum()} records with mfg data")
print(f"   Procurement: {final_data['Procurement_Orders_Count'].notna().sum()} records with procurement data")

print(f"\nFINAL DATASET COLUMNS ({len(final_data.columns)} total):")
for i, col in enumerate(final_data.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\nEnhanced merge completed by: Shoumika Anannyo")
print("Dataset ready for comprehensive analysis!")

# ==========================================
# DIAGNOSTIC CODE FOR PROCUREMENT LINKAGE
# Author: Shoumika Anannyo
# ==========================================

print("\n" + "="*60)
print("DIAGNOSTIC ANALYSIS: PROCUREMENT LINKAGE ISSUE")
print("="*60)

# Check Material_Code matching between datasets
print("\nMATERIAL_CODE ANALYSIS:")
print(f"Unique Material_Codes in Manufacturing: {mfg['Material_Code'].nunique()}")
print(f"Unique Material_Codes in Procurement: {proc['Material_Code'].nunique()}")

# Check for overlap between manufacturing and procurement
manufacturing_codes = set(mfg['Material_Code'].dropna().unique())
procurement_codes = set(proc['Material_Code'].dropna().unique())
overlap = manufacturing_codes.intersection(procurement_codes)

print(f"Overlapping Material_Codes: {len(overlap)}")
if overlap:
    print("Sample overlapping codes:", list(overlap)[:5])
else:
    print("NO OVERLAPPING CODES FOUND")

# Check what's actually in each dataset
print(f"\nSAMPLE CODES FROM EACH DATASET:")
print("Manufacturing Material_Codes (sample):", list(manufacturing_codes)[:5])
print("Procurement Material_Codes (sample):", list(procurement_codes)[:5])

# Check Product_Interest mapping
print(f"\nPRODUCT_INTEREST ANALYSIS:")
print(f"Unique Product_Interest values in Sales: {sales['Product_Interest'].nunique()}")
print("Sample Product_Interest values:", sales['Product_Interest'].dropna().unique()[:10])

# Check if Product_Interest matches Material_Name
print(f"\nPRODUCT_INTEREST → MATERIAL_NAME MAPPING:")
product_interest_values = set(sales['Product_Interest'].dropna().unique())
material_names = set(mfg['Material_Name'].dropna().unique())
interest_name_overlap = product_interest_values.intersection(material_names)

print(f"Product_Interest values that match Material_Name: {len(interest_name_overlap)}")
if interest_name_overlap:
    print("Sample matches:", list(interest_name_overlap)[:5])

# Check RawMaterial_Name in procurement
print(f"\nPROCUREMENT RAW MATERIAL ANALYSIS:")
print(f"Unique RawMaterial_Names in Procurement: {proc['RawMaterial_Name'].nunique()}")
print("Sample RawMaterial_Names:", proc['RawMaterial_Name'].dropna().unique()[:10])

# Check if we can link through RawMaterial_Name -> Material_Name
raw_materials = set(proc['RawMaterial_Name'].dropna().unique())
raw_material_overlap = raw_materials.intersection(material_names)
print(f"RawMaterial_Names that match Material_Names: {len(raw_material_overlap)}")
if raw_material_overlap:
    print("Sample RawMaterial matches:", list(raw_material_overlap)[:5])

# Data quality checks
print(f"\nDATA QUALITY CHECKS:")
print(f"Manufacturing - Missing Material_Codes: {mfg['Material_Code'].isna().sum()}")
print(f"Procurement - Missing Material_Codes: {proc['Material_Code'].isna().sum()}")
print(f"Sales - Missing Product_Interest: {sales['Product_Interest'].isna().sum()}")

# Check final merged data procurement columns
print(f"\nFINAL DATASET PROCUREMENT STATUS:")
procurement_cols = [col for col in final_data.columns if any(x in col.lower() for x in ['procurement', 'supplier', 'qty_mt', 'unit_cost', 'co2'])]
print(f"Procurement-related columns in final data: {procurement_cols}")
for col in procurement_cols:
    non_null_count = final_data[col].notna().sum()
    print(f"  - {col}: {non_null_count} non-null values ({non_null_count/len(final_data)*100:.1f}%)")

print("\n" + "="*60)
print("SUGGESTED SOLUTIONS:")
print("1. If Material_Codes don't match, check if there's a mapping table")
print("2. Try linking through RawMaterial_Name if it matches Material_Name")
print("3. Check if Product_Interest can map to RawMaterial_Name")
print("4. Look for common patterns in the code structures")
print("="*60)

# ==========================================
# FIXED PROCUREMENT MERGE WITH RAW MATERIAL MAPPING
# Author: Shoumika Anannyo
# ==========================================

print("\n" + "="*60)
print("CREATING RAW MATERIAL TO FINISHED PRODUCT MAPPING")
print("="*60)

# Since we don't have a direct mapping, we'll create logical relationships
# based on product categories and material types

# Create a mapping logic based on product categories
category_raw_material_map = {
    'Packaging Materials': ['Sugarcane Fiber', 'Wood Pulp', 'Cassava Starch', 'Corn Starch'],
    'Consumer & Home Care': ['Palm Oil Derivative', 'Algae Oil', 'Soy Protein Extract'],
    'Automotive Solutions': ['Algae Oil', 'Palm Oil Derivative'],
    'Specialty Polymers': ['Cassava Starch', 'Corn Starch', 'Soy Protein Extract'],
    'Industrial Lubricants': ['Algae Oil', 'Palm Oil Derivative']
}

# Create a mapping dataset
mapping_data = []
for category, raw_materials in category_raw_material_map.items():
    for raw_material in raw_materials:
        mapping_data.append({'Category': category, 'RawMaterial_Name': raw_material})

category_mapping_df = pd.DataFrame(mapping_data)
print("Created category-based raw material mapping")

# Merge procurement with category mapping
proc_with_category = proc.merge(category_mapping_df, how='left', on='RawMaterial_Name')
print(f"Procurement with category mapping: {proc_with_category.shape}")

# Now aggregate procurement by category (instead of Material_Code)
proc_agg_by_category = proc_with_category.groupby('Category').agg({
    'Qty_MT': 'sum',
    'Unit_Cost_USD': 'mean',
    'CO2_Emissions_kg_MT': 'mean',
    'Supplier_Name': lambda x: ', '.join(x.unique()),
    'PO_Number': 'count',
    'On_Time_Y_N': lambda x: (x == 'Y').mean() * 100,
    'RawMaterial_Name': lambda x: ', '.join(x.unique())
}).reset_index()

proc_agg_by_category.rename(columns={
    'PO_Number': 'Procurement_Orders_Count',
    'On_Time_Y_N': 'On_Time_Delivery_Percent',
    'RawMaterial_Name': 'RawMaterials_Used'
}, inplace=True)

print(f"Procurement aggregated by category: {proc_agg_by_category.shape}")

# Merge this category-based procurement data with our main dataset
print("\nMERGING PROCUREMENT DATA BY CATEGORY...")
final_data_with_procurement = final_data.merge(
    proc_agg_by_category, 
    how='left', 
    on='Category', 
    suffixes=('', '_Proc')
)

print(f"Final dataset with procurement: {final_data_with_procurement.shape}")

# Check procurement linkage success
procurement_linked = final_data_with_procurement['Procurement_Orders_Count'].notna().sum()
print(f"Procurement linkage success: {procurement_linked} records ({procurement_linked/len(final_data_with_procurement)*100:.1f}%)")

# Export the enhanced dataset
enhanced_output_path = os.path.join(save_dir, "GreenFuture_Enhanced_Master_With_Procurement.xlsx")
final_data_with_procurement.to_excel(enhanced_output_path, index=False)

print(f"\nENHANCED DATASET WITH PROCUREMENT SAVED AT:")
print(f"   {enhanced_output_path}")

print(f"\nFINAL DATASET OVERVIEW:")
print(f"   Total records: {len(final_data_with_procurement)}")
print(f"   Total columns: {len(final_data_with_procurement.columns)}")
print(f"   Procurement data linked: {procurement_linked} records")

# Show procurement columns that now have data
procurement_cols = [col for col in final_data_with_procurement.columns 
                   if any(x in col.lower() for x in ['procurement', 'supplier', 'qty_mt', 'unit_cost', 'co2', 'rawmaterial'])]

print(f"\nPROCUREMENT DATA NOW AVAILABLE:")
for col in procurement_cols:
    non_null_count = final_data_with_procurement[col].notna().sum()
    if non_null_count > 0:
        print(f"   {col}: {non_null_count} records ({non_null_count/len(final_data_with_procurement)*100:.1f}%)")

print(f"\nEnhanced merge completed by: Shoumika Anannyo")
print("Procurement data now linked through product category mapping!")


import matplotlib.pyplot as plt
import numpy as np

# Data from the image
supply_chain_cost = 452
other_cost = 1124
total_cost = supply_chain_cost + other_cost

# Cost components
cost_components = {
    'Raw Materials': 152,
    'Processing': 1276,
    'Other': 148
}

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Supply Chain Cost Analysis', fontsize=16, fontweight='bold')

# Pie chart for cost distribution
labels = ['Supply Chain Costs', 'Other Manufacturing Costs']
sizes = [supply_chain_cost, other_cost]
colors = ['#ff6b6b', '#4ecdc4']
explode = (0.1, 0)  # explode the supply chain slice

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Cost Distribution\n(Supply Chain vs Other Costs)')

# Bar chart for cost components
components = list(cost_components.keys())
values = list(cost_components.values())
bar_colors = ['#ff9e6d', '#6a89cc', '#b8e994']

bars = ax2.bar(components, values, color=bar_colors)
ax2.set_title('Cost Components per Metric Ton')
ax2.set_ylabel('Cost ($/MT)')

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'${value}/MT', ha='center', va='bottom')

# Key insights
insights = [
    f"• Supply chain costs represent {supply_chain_cost/total_cost*100:.1f}% of total manufacturing costs",
    f"• Every ${supply_chain_cost} in raw materials generates ${total_cost} in total manufacturing costs",
    f"• 1,581,462 MT raw materials support 4,218,157 MT production"
]

ax3.text(0.1, 0.8, "KEY INSIGHTS:", fontsize=14, fontweight='bold')
for i, insight in enumerate(insights):
    ax3.text(0.1, 0.7 - i*0.15, insight, fontsize=12)

ax3.axis('off')  # Hide axes for the insights panel

# Add total cost display
ax3.text(0.1, 0.3, f"Total Manufacturing Cost: ${total_cost}/MT", 
         fontsize=14, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


# ===============================================================
# SECTION IV: DIAGNOSTIC ANALYTICS – WHY IS IT HAPPENING?
# ===============================================================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------------------
# 1 LOAD MANUFACTURING DATA
#---------------------------------------------
path_mfg = "/Users/shoumikaanannyo/Downloads/ManufacturingProductionAnalysis.xlsx"
mfg = pd.read_excel(path_mfg)

#---------------------------------------------
# 2 OUTPUT DIRECTORY - YOUR DESKTOP
#---------------------------------------------
desktop_path = "/Users/shoumikaanannyo/Desktop"
out_dir = os.path.join(desktop_path, "GreenFuture_Diagnostic_Analysis")
os.makedirs(out_dir, exist_ok=True)

#---------------------------------------------
# 3 LATEX-STYLED VISUAL CONFIG
#---------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
sns.set_theme(style="whitegrid")
palette_research = ["#1B263B", "#415A77", "#778DA9", "#E0E1DD"]
sns.set_style({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "#B0B0B0",
    "axes.spines.right": False,
    "axes.spines.top": False
})

#---------------------------------------------
# 4 CREATE DERIVED METRICS
#---------------------------------------------
# Use your actual column names from the manufacturing data
mfg["Cost_Variance_$"] = mfg["Actual_Cost_per_MT ($)"] - mfg["Std_Cost_per_MT ($)"]
mfg["Cost_Variance_%"] = (mfg["Cost_Variance_$"] / mfg["Std_Cost_per_MT ($)"]) * 100
mfg["Efficiency_Ratio"] = mfg["Actual_Quantity (MT)"] / mfg["Planned_Quantity (MT)"]

#---------------------------------------------
# 5 SUMMARY STATS BY PLANT
#---------------------------------------------
summary = mfg.groupby("Plant_Code").agg(
    Yield_mean=("Yield (%)", "mean"),
    Yield_std=("Yield (%)", "std"),
    CostVar_mean=("Cost_Variance_%", "mean"),
    CostVar_std=("Cost_Variance_%", "std"),
    EffRatio_mean=("Efficiency_Ratio", "mean")
).reset_index()

print("\n===== Plant-Level Diagnostic Summary =====")
print(summary.round(2))

#---------------------------------------------
# 6 PEARSON CORRELATION MATRIX
#---------------------------------------------
corr_vars = ["Yield (%)", "Std_Cost_per_MT ($)", "Actual_Cost_per_MT ($)",
             "Cost_Variance_$", "Cost_Variance_%", "Efficiency_Ratio"]
corr_matrix = mfg[corr_vars].corr(method="pearson")
print("\n===== Pearson Correlation Matrix =====")
print(corr_matrix.round(3))

#---------------------------------------------
# 7 BOX PLOT — YIELD BY PLANT
#---------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Plant_Code", y="Yield (%)", data=mfg, palette=palette_research)
plt.title("Yield (%) Distribution by Plant")
plt.xlabel("Plant Code")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.savefig(f"{out_dir}/IV1_Yield_byPlant.png", dpi=300)
plt.close()

#---------------------------------------------
# 8 SCATTER — YIELD VS COST VARIANCE (%)
#---------------------------------------------
plt.figure(figsize=(5.8,4.2))
sns.regplot(x="Yield (%)", y="Cost_Variance_%", data=mfg,
            scatter_kws={"alpha":0.6, "s":30, "color":palette_research[1]},
            line_kws={"color":palette_research[0], "lw":1.2})
plt.title("Yield vs Cost Variance (%)")
plt.xlabel("Yield (%)")
plt.ylabel("Cost Variance (%)")
r_val = mfg["Yield (%)"].corr(mfg["Cost_Variance_%"])
plt.text(0.05, 0.95, f"r = {r_val:.2f}", transform=plt.gca().transAxes,
         ha="left", va="top", fontsize=10, fontstyle="italic")
plt.tight_layout()
plt.savefig(f"{out_dir}/IV2_Yield_vs_CostVariance.png", dpi=300)
plt.close()

#---------------------------------------------
# 9 BAR — AVG COST VARIANCE (%) BY PLANT
#---------------------------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="Plant_Code", y="CostVar_mean", data=summary, palette=palette_research)
plt.title("Average Cost Variance (%) by Plant")
plt.xlabel("Plant Code")
plt.ylabel("Mean Cost Variance (%)")
plt.tight_layout()
plt.savefig(f"{out_dir}/IV3_CostVariance_byPlant.png", dpi=300)
plt.close()

#---------------------------------------------
# 10 HEATMAP — CORRELATION MATRIX
#---------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="crest", fmt=".2f", linewidths=0.3,
            cbar_kws={"label":"Pearson r"})
plt.title("Correlation Matrix — Production Efficiency Drivers")
plt.tight_layout()
plt.savefig(f"{out_dir}/IV4_CorrelationMatrix.png", dpi=300)
plt.close()

print(f"\nDiagnostic visuals saved to: {out_dir}")
print(f"Location: {out_dir}")


# ===============================================================
# SUPPLY CHAIN VS MANUFACTURING COST ANALYSIS
# ===============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_cost_breakdown_analysis():
    print("CREATING SUPPLY CHAIN VS MANUFACTURING COST ANALYSIS...")
    
    # Load datasets
    mfg_path = '/Users/shoumikaanannyo/Downloads/ManufacturingProductionAnalysis.xlsx'
    sc_path = '/Users/shoumikaanannyo/Downloads/SupplyChainAnalysis.xlsx'
    
    mfg = pd.read_excel(mfg_path)
    sc = pd.read_excel(sc_path)
    
    print(f"Manufacturing data: {mfg.shape}")
    print(f"Supply chain data: {sc.shape}")
    
    # Calculate cost metrics
    # Manufacturing cost per MT (already in dataset)
    mfg_cost_per_mt = mfg['Actual_Cost_per_MT ($)'].mean()
    
    # Supply chain cost per MT (average unit cost)
    sc_cost_per_mt = sc['Unit_Cost ($)'].mean()
    
    # Calculate procurement portion of manufacturing costs
    procurement_portion = (sc_cost_per_mt / mfg_cost_per_mt) * 100
    other_costs = 100 - procurement_portion
    
    # Calculate total volumes for context
    total_mfg_volume = mfg['Actual_Quantity (MT)'].sum()
    total_sc_volume = sc['Qty (MT)'].sum()
    
    print(f"CALCULATED METRICS:")
    print(f"   Manufacturing Cost: ${mfg_cost_per_mt:.2f}/MT")
    print(f"   Supply Chain Cost: ${sc_cost_per_mt:.2f}/MT")
    print(f"   Procurement Portion: {procurement_portion:.1f}% of manufacturing costs")
    print(f"   Total Manufacturing Volume: {total_mfg_volume:,.0f} MT")
    print(f"   Total Procurement Volume: {total_sc_volume:,.0f} MT")
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Professional styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "figure.dpi": 300,
    })
    
    colors = ['#DC143C', '#4169E1', '#2E8B57', '#FF8C00']
    
    # =====================
    # PANEL 1: COST BREAKDOWN PIE CHART
    # =====================
    cost_breakdown = [procurement_portion, other_costs]
    labels = [f'Supply Chain Costs\n${sc_cost_per_mt:.0f}/MT', f'Other Manufacturing\n${mfg_cost_per_mt - sc_cost_per_mt:.0f}/MT']
    
    wedges, texts, autotexts = ax1.pie(cost_breakdown, labels=labels, colors=[colors[0], colors[1]],
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Make the supply chain wedge stand out
    wedges[0].set_edgecolor('white')
    wedges[0].set_linewidth(2)
    
    ax1.set_title('MANUFACTURING COST BREAKDOWN\n(Supply Chain vs Other Costs)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # =====================
    # PANEL 2: COST COMPARISON BAR CHART
    # =====================
    categories = ['Supply Chain', 'Other Costs', 'Total Manufacturing']
    costs = [sc_cost_per_mt, mfg_cost_per_mt - sc_cost_per_mt, mfg_cost_per_mt]
    bar_colors = [colors[0], colors[1], colors[2]]
    
    bars = ax2.bar(categories, costs, color=bar_colors, alpha=0.8, width=0.7)
    ax2.set_ylabel('Cost per MT ($)', fontweight='bold', fontsize=12)
    ax2.set_title('COST COMPONENTS PER METRIC TON', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 20,
                f'${cost:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
        
        # Add percentage for first two bars
        if i < 2:
            percentage = (cost / mfg_cost_per_mt) * 100
            ax2.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{percentage:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=10, color='white')
    
    # Add total cost annotation
    ax2.text(0.5, 0.95, f'Total Manufacturing Cost: ${mfg_cost_per_mt:.0f}/MT', 
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=colors[2],
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # =====================
    # MAIN TITLE AND INSIGHTS
    # =====================
    plt.suptitle('SUPPLY CHAIN COST IMPACT ON MANUFACTURING', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Key insights box
    insight_text = (
        f"KEY INSIGHTS:\n"
        f"• Supply chain costs represent {procurement_portion:.1f}% of total manufacturing costs\n"
        f"• Every ${sc_cost_per_mt:.0f} in raw materials generates ${mfg_cost_per_mt:.0f} in total manufacturing costs\n"
        f"• {total_sc_volume:,.0f} MT raw materials support {total_mfg_volume:,.0f} MT production"
    )
    
    fig.text(0.5, 0.02, insight_text, ha='center', va='bottom', 
            fontsize=12, fontweight='bold', linespacing=1.5,
            bbox=dict(boxstyle="round,pad=1.0", facecolor='#F0F8FF', edgecolor='#4169E1', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # =====================
    # SAVE TO DESKTOP
    # =====================
    desktop_path = str(Path.home() / "Desktop")
    output_path = f"{desktop_path}/Supply_Chain_Cost_Analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nCOST ANALYSIS VISUAL CREATED!")
    print(f"Saved to: {output_path}")
    
    # Return metrics for potential use in report
    return {
        'mfg_cost_per_mt': mfg_cost_per_mt,
        'sc_cost_per_mt': sc_cost_per_mt,
        'procurement_portion': procurement_portion,
        'total_mfg_volume': total_mfg_volume,
        'total_sc_volume': total_sc_volume
    }

# Run the analysis
if __name__ == "__main__":
    results = create_cost_breakdown_analysis()


    # ===============================================================
# PREDICTIVE ANALYSIS FOR STRATEGIC RECOMMENDATIONS - SIMPLIFIED
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def predictive_analysis_simple():
    print("RUNNING PREDICTIVE ANALYSIS FOR RECOMMENDATIONS...")
    
    # Load datasets
    mfg_path = '/Users/shoumikaanannyo/Downloads/ManufacturingProductionAnalysis.xlsx'
    sc_path = '/Users/shoumikaanannyo/Downloads/SupplyChainAnalysis.xlsx'
    
    mfg = pd.read_excel(mfg_path)
    sc = pd.read_excel(sc_path)
    
    print(f"Data loaded: MFG({mfg.shape}), SC({sc.shape})")
    
    # ===============================================================
    # RECOMMENDATION 1: COST MODEL REVISION
    # ===============================================================
    print("\nRECOMMENDATION 1: COST MODEL REVISION")
    
    # Current cost overrun calculation
    current_std_cost = mfg['Std_Cost_per_MT ($)'].mean()
    current_actual_cost = mfg['Actual_Cost_per_MT ($)'].mean()
    cost_overrun_pct = ((current_actual_cost - current_std_cost) / current_std_cost) * 100
    cost_overrun_amt = current_actual_cost - current_std_cost
    
    # Total production volume from manufacturing
    total_production_volume = mfg['Actual_Quantity (MT)'].sum()
    current_annual_overspend = total_production_volume * cost_overrun_amt
    
    print(f"Current Cost Analysis:")
    print(f"   Standard Cost: ${current_std_cost:.0f}/MT")
    print(f"   Actual Cost: ${current_actual_cost:.0f}/MT")
    print(f"   Overrun: {cost_overrun_pct:.1f}% (${cost_overrun_amt:.0f}/MT)")
    print(f"   Annual Overspend: ${current_annual_overspend:,.0f}")
    
    # Manual correlation calculation (simplified)
    cost_correlation = mfg['Std_Cost_per_MT ($)'].corr(mfg['Actual_Cost_per_MT ($)'])
    print(f"Cost Model Accuracy:")
    print(f"   Standard vs Actual Cost Correlation: {cost_correlation:.3f}")
    print(f"   Model explains {cost_correlation**2*100:.1f}% of cost variance")
    
    # Projected savings with improved model (assuming 80% accuracy improvement)
    projected_overspend_reduction = current_annual_overspend * 0.8
    print(f"Projected Savings with Improved Cost Model:")
    print(f"   Estimated Annual Savings: ${projected_overspend_reduction:,.0f}")
    
    # ===============================================================
    # RECOMMENDATION 2: YIELD IMPROVEMENT PROGRAM
    # ===============================================================
    print("\nRECOMMENDATION 2: YIELD IMPROVEMENT PROGRAM")
    
    # Current yield analysis
    current_yield = mfg['Yield (%)'].mean()
    industry_benchmark = 95.0
    yield_gap = industry_benchmark - current_yield
    
    # Plant-specific analysis
    plant_yield = mfg.groupby('Plant_Code')['Yield (%)'].mean().sort_values()
    worst_plant = plant_yield.index[0]
    worst_plant_yield = plant_yield.iloc[0]
    best_plant_yield = plant_yield.iloc[-1]
    improvement_potential = best_plant_yield - worst_plant_yield
    
    print(f"Current Yield Analysis:")
    print(f"   Average Yield: {current_yield:.1f}%")
    print(f"   Industry Benchmark: {industry_benchmark:.1f}%")
    print(f"   Yield Gap: {yield_gap:.1f}%")
    print(f"   Worst Plant ({worst_plant}): {worst_plant_yield:.1f}%")
    print(f"   Best Plant: {best_plant_yield:.1f}%")
    print(f"   Improvement Potential: {improvement_potential:.1f}%")
    
    # Financial impact calculation
    raw_material_cost = sc['Unit_Cost ($)'].mean()
    worst_plant_volume = mfg[mfg['Plant_Code'] == worst_plant]['Actual_Quantity (MT)'].sum()
    
    # Savings from closing gap to industry benchmark
    material_savings_benchmark = (yield_gap/100) * raw_material_cost * total_production_volume
    
    # Savings from worst plant improvement
    material_savings_worst_plant = (improvement_potential/100) * raw_material_cost * worst_plant_volume
    
    print(f"Yield Improvement Financial Impact:")
    print(f"   Raw Material Cost: ${raw_material_cost:.0f}/MT")
    print(f"   Total Production: {total_production_volume:,.0f} MT")
    print(f"   {worst_plant} Production: {worst_plant_volume:,.0f} MT")
    print(f"   Closing to Industry Benchmark Savings: ${material_savings_benchmark:,.0f}")
    print(f"   Worst Plant Improvement Savings: ${material_savings_worst_plant:,.0f}")
    
    # ===============================================================
    # RECOMMENDATION 3: PORTFOLIO RATIONALIZATION
    # ===============================================================
    print("\nRECOMMENDATION 3: PORTFOLIO RATIONALIZATION")
    
    # Analyze product complexity impact
    if 'Category' in mfg.columns:
        # Calculate complexity metrics
        skus_per_category = mfg.groupby('Category').size()
        avg_yield_by_category = mfg.groupby('Category')['Yield (%)'].mean()
        avg_cost_by_category = mfg.groupby('Category')['Actual_Cost_per_MT ($)'].mean()
        
        complexity_analysis = pd.DataFrame({
            'SKU_Count': skus_per_category,
            'Avg_Yield': avg_yield_by_category,
            'Avg_Cost': avg_cost_by_category
        })
        
        # Calculate complexity-performance correlation
        complexity_corr = complexity_analysis['SKU_Count'].corr(complexity_analysis['Avg_Yield'])
        cost_complexity_corr = complexity_analysis['SKU_Count'].corr(complexity_analysis['Avg_Cost'])
        
        print(f"Portfolio Complexity Analysis:")
        print(f"   SKU Count - Yield Correlation: {complexity_corr:.3f}")
        print(f"   SKU Count - Cost Correlation: {cost_complexity_corr:.3f}")
        
        # Identify underperforming categories
        complexity_analysis['Performance_Score'] = (complexity_analysis['Avg_Yield'] / 100) * (1 / complexity_analysis['Avg_Cost'])
        bottom_performers = complexity_analysis.nsmallest(2, 'Performance_Score')
        
        print(f"Bottom Performing Categories:")
        for category, row in bottom_performers.iterrows():
            print(f"   {category}: {row['SKU_Count']} SKUs, {row['Avg_Yield']:.1f}% yield, ${row['Avg_Cost']:.0f}/MT")
    
    # Estimate savings from portfolio simplification
    # Assumption: 20% SKU reduction improves efficiency by 5%
    estimated_efficiency_gain = 0.05
    current_total_costs = total_production_volume * current_actual_cost
    portfolio_simplification_savings = current_total_costs * estimated_efficiency_gain
    
    print(f"Portfolio Rationalization Impact:")
    print(f"   Current Total Manufacturing Costs: ${current_total_costs:,.0f}")
    print(f"   Estimated Efficiency Gain: {estimated_efficiency_gain*100:.1f}%")
    print(f"   Projected Annual Savings: ${portfolio_simplification_savings:,.0f}")
    
    # ===============================================================
    # SUMMARY OF PREDICTED IMPACTS
    # ===============================================================
    print("\n" + "="*60)
    print("PREDICTED FINANCIAL IMPACT SUMMARY")
    print("="*60)
    
    summary_data = {
        'Recommendation': [
            '1. Cost Model Revision',
            '2. Yield Improvement', 
            '3. Portfolio Rationalization',
            'TOTAL POTENTIAL IMPACT'
        ],
        'Annual Savings': [
            projected_overspend_reduction,
            material_savings_worst_plant,
            portfolio_simplification_savings,
            projected_overspend_reduction + material_savings_worst_plant + portfolio_simplification_savings
        ],
        'Key Metric': [
            f"Eliminate {cost_overrun_pct:.1f}% cost overrun",
            f"Improve {worst_plant} yield by {improvement_potential:.1f}%",
            "20% SKU reduction, 5% efficiency gain",
            "Combined strategic impact"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Create visualization of impacts
    create_impact_visualization(summary_df, total_production_volume, current_annual_overspend)
    
    return summary_df

def create_impact_visualization(summary_df, total_volume, current_overspend):
    """Create visualization of predicted impacts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors
    colors = ['#2E8B57', '#4169E1', '#FF8C00', '#1B263B']
    
    # Plot 1: Savings by recommendation
    recommendations = summary_df['Recommendation'][:3]
    savings = summary_df['Annual Savings'][:3]
    
    bars = ax1.bar(recommendations, savings, color=colors[:3], alpha=0.8)
    ax1.set_ylabel('Annual Savings ($)', fontweight='bold', fontsize=12)
    ax1.set_title('PREDICTED ANNUAL SAVINGS BY RECOMMENDATION', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, saving in zip(bars, savings):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000000,
                f'${saving/1000000:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Impact comparison
    categories = ['Current Overspend', 'Total Potential Savings']
    values = [current_overspend, summary_df['Annual Savings'].iloc[3]]
    
    bars = ax2.bar(categories, values, color=['#DC143C', '#2E8B57'], alpha=0.8)
    ax2.set_ylabel('Amount ($)', fontweight='bold', fontsize=12)
    ax2.set_title('CURRENT VS POTENTIAL FINANCIAL IMPACT', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and percentage
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000000,
                f'${value/1000000:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add percentage improvement
    improvement_pct = ((summary_df['Annual Savings'].iloc[3] - current_overspend) / current_overspend) * 100
    ax2.text(0.5, 0.8, f'Potential Improvement: {improvement_pct:+.1f}%', 
            transform=ax2.transAxes, ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save to desktop
    output_path = f"{Path.home()}/Desktop/Predictive_Impact_Analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nImpact visualization saved to: {output_path}")
    
    plt.show()

# Run the predictive analysis
if __name__ == "__main__":
    results = predictive_analysis_simple()