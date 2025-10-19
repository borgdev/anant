# Practical Example: Dataset Analysis with "anant"

**Document Version**: 1.0  
**Date**: October 17, 2025  
**Purpose**: Demonstrate practical usage of enhanced "anant" library for dataset analysis

## Example: E-commerce Transaction Analysis

This example demonstrates how to use the enhanced "anant" library to analyze e-commerce transaction data stored in parquet files, focusing on customer behavior, product relationships, and transaction patterns.

### Dataset Structure

Our example dataset consists of three main parquet files:

1. **transactions.parquet** - Core transaction data (incidences)
2. **customers.parquet** - Customer properties  
3. **products.parquet** - Product properties

### Sample Data Setup

```python
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from anant import DatasetAnalyzer, DatasetHypergraph

# Create sample transaction data
def create_sample_ecommerce_data():
    """Create sample e-commerce dataset for demonstration"""
    
    # Generate sample transactions (incidences)
    np.random.seed(42)
    n_customers = 1000
    n_products = 500
    n_transactions = 10000
    
    # Generate transaction incidences
    customer_ids = [f"customer_{i}" for i in range(n_customers)]
    product_ids = [f"product_{i}" for i in range(n_products)]
    
    transactions = []
    for _ in range(n_transactions):
        customer = np.random.choice(customer_ids)
        product = np.random.choice(product_ids)
        
        # Create hyperedge for each transaction
        transaction_id = f"transaction_{len(transactions)}"
        
        # Customer-transaction incidence
        transactions.append({
            "edge_id": transaction_id,
            "node_id": customer,
            "weight": np.random.exponential(50),  # Purchase amount
            "role": "buyer",
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
        
        # Product-transaction incidence  
        transactions.append({
            "edge_id": transaction_id,
            "node_id": product,
            "weight": np.random.poisson(2) + 1,  # Quantity
            "role": "product",
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
    
    transaction_df = pl.DataFrame(transactions)
    
    # Generate customer properties
    customers = []
    for customer_id in customer_ids:
        customers.append({
            "uid": customer_id,
            "age": np.random.randint(18, 80),
            "income_bracket": np.random.choice(["low", "medium", "high"]),
            "location": np.random.choice(["urban", "suburban", "rural"]),
            "loyalty_score": np.random.uniform(0, 1),
            "registration_date": datetime.now() - timedelta(days=np.random.randint(30, 1000)),
            "total_purchases": np.random.poisson(10),
            "preferred_category": np.random.choice(["electronics", "clothing", "books", "home"])
        })
    
    customer_df = pl.DataFrame(customers)
    
    # Generate product properties
    products = []
    categories = ["electronics", "clothing", "books", "home", "sports"]
    for product_id in product_ids:
        products.append({
            "uid": product_id,
            "category": np.random.choice(categories),
            "price": np.random.uniform(10, 1000),
            "rating": np.random.uniform(1, 5),
            "stock_level": np.random.randint(0, 1000),
            "brand": f"brand_{np.random.randint(1, 50)}",
            "launch_date": datetime.now() - timedelta(days=np.random.randint(1, 2000)),
            "is_seasonal": np.random.choice([True, False]),
            "discount_rate": np.random.uniform(0, 0.5)
        })
    
    product_df = pl.DataFrame(products)
    
    return transaction_df, customer_df, product_df

# Save sample data to parquet files
def save_sample_data():
    transactions, customers, products = create_sample_ecommerce_data()
    
    # Save to parquet files
    transactions.write_parquet("sample_data/transactions.parquet")
    customers.write_parquet("sample_data/customers.parquet") 
    products.write_parquet("sample_data/products.parquet")
    
    print("Sample data saved to parquet files")
    return transactions, customers, products
```

### Load and Analyze Dataset

```python
def comprehensive_ecommerce_analysis():
    """
    Comprehensive analysis of e-commerce dataset using anant
    """
    
    # Load dataset from parquet files
    print("Loading e-commerce dataset...")
    analyzer = DatasetAnalyzer.from_parquet_dataset(
        dataset_path="sample_data/",
        incidence_file="transactions.parquet",
        node_props_file="customers.parquet",
        edge_props_file="products.parquet",
        lazy_loading=True
    )
    
    # === Basic Dataset Overview ===
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    incidence_stats = analyzer.analyze_incidence_patterns()
    print(f"Total customers: {incidence_stats['unique_nodes']:,}")
    print(f"Total transactions: {incidence_stats['unique_edges']:,}")
    print(f"Total incidences: {incidence_stats['incidence_count']:,}")
    print(f"Average customer activity: {incidence_stats['node_degree_stats']['mean_degree']:.2f}")
    
    # === Weight Analysis (Purchase Behavior) ===
    print("\n" + "="*50) 
    print("PURCHASE BEHAVIOR ANALYSIS")
    print("="*50)
    
    weight_analysis = analyzer.analyze_weight_distributions()
    
    print("Customer Purchase Patterns:")
    print(f"  Average purchase amount: ${weight_analysis['node_weights']['mean']:.2f}")
    print(f"  Purchase amount range: ${weight_analysis['node_weights']['min']:.2f} - ${weight_analysis['node_weights']['max']:.2f}")
    print(f"  Purchase variability (std): ${weight_analysis['node_weights']['std']:.2f}")
    
    print("\nProduct Sales Patterns:")  
    print(f"  Average product quantity: {weight_analysis['edge_weights']['mean']:.2f}")
    print(f"  Quantity range: {weight_analysis['edge_weights']['min']:.0f} - {weight_analysis['edge_weights']['max']:.0f}")
    
    # === Customer Segmentation Analysis ===
    print("\n" + "="*50)
    print("CUSTOMER SEGMENTATION")
    print("="*50)
    
    centrality_analysis = analyzer.weighted_centrality_analysis()
    
    # High-value customers
    vip_customers = (
        centrality_analysis["weighted_degree"]
        .filter(pl.col("normalized_weighted_degree") > 0.9)
        .sort("weighted_degree", descending=True)
        .head(10)
    )
    
    print("Top 10 VIP Customers (by total purchase amount):")
    print(vip_customers.select(["node_id", "weighted_degree", "unweighted_degree"]))
    
    # Customer behavior segments
    customer_segments = (
        centrality_analysis["weighted_degree"]
        .with_columns([
            pl.when(pl.col("normalized_weighted_degree") > 0.8)
            .then("VIP")
            .when(pl.col("normalized_weighted_degree") > 0.5)
            .then("Regular")
            .when(pl.col("normalized_weighted_degree") > 0.2)
            .then("Occasional")
            .otherwise("Rare")
            .alias("customer_segment")
        ])
        .group_by("customer_segment")
        .agg([
            pl.count().alias("count"),
            pl.col("weighted_degree").mean().alias("avg_purchase_amount"),
            pl.col("unweighted_degree").mean().alias("avg_transaction_count")
        ])
    )
    
    print("\nCustomer Segments:")
    print(customer_segments)
    
    # === Product Performance Analysis ===
    print("\n" + "="*50)
    print("PRODUCT PERFORMANCE")
    print("="*50)
    
    product_performance = (
        centrality_analysis["edge_centrality"]
        .sort("total_edge_weight", descending=True)
        .head(10)
    )
    
    print("Top 10 Best-Selling Products (by total quantity):")
    print(product_performance.select(["edge_id", "total_edge_weight", "edge_size"]))
    
    # === Property Correlation Analysis ===
    print("\n" + "="*50)
    print("PROPERTY CORRELATION ANALYSIS")
    print("="*50)
    
    # Analyze customer property correlations
    customer_numeric_props = ["age", "loyalty_score", "total_purchases"]
    customer_correlations = analyzer.property_correlation_analysis(
        customer_numeric_props, 
        level="nodes"
    )
    
    print("Customer Property Correlations:")
    print(customer_correlations)
    
    # === Temporal Analysis ===
    print("\n" + "="*50)
    print("TEMPORAL ANALYSIS")
    print("="*50)
    
    temporal_analysis = analyzer.temporal_analysis(
        time_column="timestamp",
        level="incidences", 
        time_window="1w"  # Weekly aggregation
    )
    
    print(f"Analysis timespan: {temporal_analysis['time_range']['span']}")
    print(f"Active weeks: {temporal_analysis['activity_periods']}")
    
    # Weekly sales trends
    weekly_sales = temporal_analysis["temporal_stats"].head(10)
    print("\nWeekly Sales Trends (first 10 weeks):")
    print(weekly_sales)
    
    # === Advanced Analysis: Customer-Product Affinity ===
    print("\n" + "="*50)
    print("CUSTOMER-PRODUCT AFFINITY ANALYSIS")
    print("="*50)
    
    # Find customers with similar purchasing patterns
    customer_product_matrix = (
        analyzer.hg.incidences._data
        .filter(pl.col("role") == "buyer")
        .group_by(["node_id", "edge_id"])
        .agg([
            pl.col("weight").sum().alias("total_spent")
        ])
        .pivot(
            index="node_id",
            columns="edge_id", 
            values="total_spent"
        )
        .fill_null(0)
    )
    
    print(f"Customer-Product Matrix Shape: {customer_product_matrix.shape}")
    
    # === Category Analysis ===
    print("\n" + "="*50)
    print("CATEGORY PERFORMANCE")
    print("="*50)
    
    # Join product properties to analyze by category
    category_performance = (
        analyzer.hg.incidences._data
        .filter(pl.col("role") == "product")
        .join(
            analyzer.hg.edges._data.select(["uid", "category"]),
            left_on="node_id",
            right_on="uid"
        )
        .group_by("category")
        .agg([
            pl.col("weight").sum().alias("total_quantity"),
            pl.col("weight").count().alias("transaction_count"),
            pl.col("edge_id").n_unique().alias("unique_products")
        ])
        .sort("total_quantity", descending=True)
    )
    
    print("Category Performance:")
    print(category_performance)
    
    return analyzer

def advanced_recommendation_analysis(analyzer: DatasetAnalyzer):
    """
    Advanced recommendation analysis using hypergraph structure
    """
    
    print("\n" + "="*50)
    print("RECOMMENDATION ANALYSIS")
    print("="*50)
    
    # Customer similarity based on shared transactions
    customer_similarity = (
        analyzer.hg.incidences._data
        .filter(pl.col("role") == "buyer")
        .join(
            analyzer.hg.incidences._data.filter(pl.col("role") == "buyer"),
            on="edge_id",
            suffix="_other"
        )
        .filter(pl.col("node_id") != pl.col("node_id_other"))
        .group_by(["node_id", "node_id_other"])
        .agg([
            pl.count().alias("shared_transactions"),
            pl.col("weight").corr(pl.col("weight_other")).alias("spending_correlation")
        ])
        .filter(pl.col("shared_transactions") >= 3)  # At least 3 shared transactions
        .sort("spending_correlation", descending=True)
    )
    
    print("Customer Similarity (top 10 pairs):")
    print(customer_similarity.head(10))
    
    # Product co-occurrence analysis
    product_cooccurrence = (
        analyzer.hg.incidences._data
        .filter(pl.col("role") == "product")
        .join(
            analyzer.hg.incidences._data.filter(pl.col("role") == "product"),
            on="edge_id",
            suffix="_other"
        )
        .filter(pl.col("node_id") != pl.col("node_id_other"))
        .group_by(["node_id", "node_id_other"])
        .agg([
            pl.count().alias("cooccurrence_count")
        ])
        .filter(pl.col("cooccurrence_count") >= 5)  # At least 5 co-occurrences
        .sort("cooccurrence_count", descending=True)
    )
    
    print("\nProduct Co-occurrence (top 10 pairs):")
    print(product_cooccurrence.head(10))
    
    return {
        "customer_similarity": customer_similarity,
        "product_cooccurrence": product_cooccurrence
    }

# Run the complete analysis
if __name__ == "__main__":
    # Generate and save sample data
    save_sample_data()
    
    # Run comprehensive analysis
    analyzer = comprehensive_ecommerce_analysis()
    
    # Run advanced recommendation analysis
    recommendations = advanced_recommendation_analysis(analyzer)
    
    # Performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    print("Analysis completed successfully!")
    print("Key capabilities demonstrated:")
    print("  ✓ Parquet file loading and processing")
    print("  ✓ Multi-level property management")
    print("  ✓ Weight-based analysis and centrality")
    print("  ✓ Incidence pattern analysis")
    print("  ✓ Temporal analysis")
    print("  ✓ Property correlation analysis") 
    print("  ✓ Customer segmentation")
    print("  ✓ Product performance analysis")
    print("  ✓ Recommendation system analysis")
```

### Key Benefits for Dataset Analysis

#### 1. **Efficient Property Storage**
- **Categorical properties**: Customer segments, product categories
- **Numerical properties**: Purchase amounts, ratings, prices
- **Temporal properties**: Registration dates, purchase timestamps
- **Vector properties**: Customer embeddings, product features

#### 2. **Weight-Based Analysis**
- **Customer value analysis**: Total spending, purchase frequency
- **Product performance**: Sales volume, transaction count
- **Relationship strength**: Customer-product affinity scores

#### 3. **Incidence Pattern Analysis**
- **Multi-role relationships**: Customers as buyers, products as items
- **Hyperedge analysis**: Transaction complexity and composition
- **Network effects**: Co-purchasing patterns, customer similarity

#### 4. **Real-time Analytics**
- **Streaming analysis**: Process large datasets in chunks
- **Lazy evaluation**: Load only necessary data portions
- **Parallel processing**: Utilize multiple cores for analysis

### Usage Scenarios

#### Social Media Analysis
```python
# Analyze user interactions on social platforms
# Nodes: Users, Posts, Hashtags
# Edges: Interactions (likes, shares, comments)
# Weights: Engagement scores, influence measures
# Properties: User demographics, post content, temporal data
```

#### Supply Chain Analysis
```python
# Analyze supply chain relationships
# Nodes: Suppliers, Products, Warehouses, Customers
# Edges: Transactions, Shipments, Orders
# Weights: Quantities, costs, delivery times
# Properties: Location data, capacity, demand patterns
```

#### Scientific Collaboration Networks
```python
# Analyze research collaboration patterns
# Nodes: Researchers, Papers, Institutions
# Edges: Collaborations, Citations, Projects
# Weights: Impact factors, collaboration strength
# Properties: Research areas, career stage, funding
```

This enhanced system provides a powerful foundation for analyzing complex datasets with rich properties and relationships, making it ideal for real-world data science applications where traditional graph analysis falls short.