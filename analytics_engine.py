import duckdb
import polars as pl
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from datetime import date

# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================
class UnitEconomicsSchema(BaseModel):
    product_subcategory: str
    total_revenue: float = Field(..., ge=0)
    total_profit: float
    net_margin_pct: float = Field(..., le=100)

class LTVCACSchema(BaseModel):
    customer_segment: str
    avg_cac: float = Field(..., ge=0)
    avg_ltv: float = Field(..., ge=0)
    ltv_to_cac_ratio: float = Field(..., ge=0)

class RFMSchema(BaseModel):
    customer_id: str
    last_purchase_date: date
    frequency: int = Field(..., gt=0)
    monetary_value: float = Field(..., ge=0)
    favorite_category: str 

class CohortSchema(BaseModel):
    cohort_month: str
    month_index: int = Field(..., ge=0)
    retention_pct: float = Field(..., ge=0, le=100)

class PromoLeakageSchema(BaseModel):
    campaign_name: str
    gross_revenue: float = Field(..., ge=0)
    total_discount_given: float = Field(..., ge=0)
    discount_leakage_pct: float = Field(..., ge=0, le=100)

class LogisticsSLASchema(BaseModel):
    warehouse: str
    avg_actual_days: float = Field(..., ge=0)
    avg_estimated_days: float = Field(..., ge=0)
    avg_delivery_lag: float
    late_deliveries: int = Field(..., ge=0)

class ReturnRootCauseSchema(BaseModel):
    return_reason: str
    total_returns: int = Field(..., ge=0)
    lost_revenue: float = Field(..., ge=0)

class MultiChannelSchema(BaseModel):
    sales_channel: str
    total_orders: int = Field(..., ge=0)
    net_revenue: float = Field(..., ge=0)
    total_ad_spend: float = Field(..., ge=0)
    roas: float = Field(..., ge=0)

class SupplierVelocitySchema(BaseModel):
    supplier: str
    total_units_sold: int = Field(..., ge=0)
    total_profit_generated: float
    avg_supplier_rating: float = Field(..., ge=0, le=5)

class VoCSentimentSchema(BaseModel):
    review_sentiment: str
    review_count: int = Field(..., ge=0)
    avg_rating: float = Field(..., ge=0, le=5)
    avg_profit_margin: float


# ==========================================
# 2. ENTERPRISE ANALYTICS ENGINE
# ==========================================
class ECommerceAnalyticsEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.sales_csv = f"{data_dir}\\ecommerce_sales_customer_analytics_150k.csv"
        self.items_csv = f"{data_dir}\\order_items.csv"
        self.catalog_csv = f"{data_dir}\\product_catalog.csv"
        self.customers_csv = f"{data_dir}\\customer_master.csv"
        
        self.conn = duckdb.connect(database=':memory:')

    def run_f1_unit_economics(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                p.product_subcategory,
                SUM(o.gross_sales) AS total_revenue,
                SUM(o.profit) AS total_profit,
                (SUM(o.profit) / NULLIF(SUM(o.gross_sales), 0)) * 100 AS net_margin_pct
            FROM '{self.items_csv}' AS o
            LEFT JOIN '{self.catalog_csv}' AS p ON o.product_id = p.product_id
            GROUP BY p.product_subcategory
            ORDER BY total_profit DESC
            LIMIT 5;
        """
        return self.conn.sql(query).pl()

    def run_f2_ltv_cac(self) -> pl.DataFrame:
        query = f"""
            WITH customer_spend AS (
                SELECT customer_id, SUM(net_sales) AS lifetime_value
                FROM '{self.sales_csv}'
                GROUP BY customer_id
            )
            SELECT 
                c.customer_segment,
                AVG(c.customer_acquisition_cost) AS avg_cac,
                AVG(s.lifetime_value) AS avg_ltv,
                AVG(s.lifetime_value) / NULLIF(AVG(c.customer_acquisition_cost), 0) AS ltv_to_cac_ratio
            FROM '{self.customers_csv}' AS c
            JOIN customer_spend AS s ON c.customer_id = s.customer_id
            GROUP BY c.customer_segment
            ORDER BY ltv_to_cac_ratio DESC;
        """
        return self.conn.sql(query).pl()

    def run_f3_rfm_segmentation(self) -> pl.DataFrame:
        query = f"""
            WITH top_customers AS (
                SELECT 
                    customer_id,
                    MAX(CAST(order_date AS DATE)) AS last_purchase_date,
                    COUNT(DISTINCT order_id) AS frequency,
                    SUM(net_sales) AS monetary_value
                FROM '{self.sales_csv}'
                GROUP BY customer_id
                ORDER BY monetary_value DESC
                LIMIT 10
            ),
            customer_categories AS (
                SELECT 
                    s.customer_id,
                    p.product_subcategory,
                    SUM(i.gross_sales) AS category_spend
                FROM '{self.sales_csv}' s
                JOIN '{self.items_csv}' i ON s.order_id = i.order_id
                JOIN '{self.catalog_csv}' p ON i.product_id = p.product_id
                WHERE s.customer_id IN (SELECT customer_id FROM top_customers)
                GROUP BY s.customer_id, p.product_subcategory
            ),
            ranked_categories AS (
                SELECT 
                    customer_id,
                    product_subcategory AS favorite_category,
                    ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY category_spend DESC) AS rank
                FROM customer_categories
            )
            SELECT 
                t.customer_id,
                t.last_purchase_date,
                t.frequency,
                t.monetary_value,
                c.favorite_category
            FROM top_customers t
            LEFT JOIN ranked_categories c ON t.customer_id = c.customer_id AND c.rank = 1
            ORDER BY t.monetary_value DESC;
        """
        return self.conn.sql(query).pl()

    # BUG FIXED: Added MIN() to properly group the cohort start date
    def run_f4_cohort_retention(self) -> pl.DataFrame:
        query = f"""
            WITH first_purchases AS (
                SELECT customer_id, DATE_TRUNC('month', MIN(CAST(order_date AS DATE))) AS cohort_month
                FROM '{self.sales_csv}'
                GROUP BY customer_id
            ),
            activity AS (
                SELECT 
                    f.cohort_month,
                    DATE_TRUNC('month', CAST(s.order_date AS DATE)) AS activity_month,
                    s.customer_id
                FROM '{self.sales_csv}' s
                JOIN first_purchases f ON s.customer_id = f.customer_id
            ),
            cohort_sizes AS (
                SELECT cohort_month, COUNT(DISTINCT customer_id) AS total_customers
                FROM first_purchases
                GROUP BY cohort_month
            ),
            retention AS (
                SELECT 
                    a.cohort_month,
                    date_diff('month', a.cohort_month, a.activity_month) AS month_index,
                    COUNT(DISTINCT a.customer_id) AS active_customers
                FROM activity a
                GROUP BY a.cohort_month, a.activity_month
            )
            SELECT 
                CAST(r.cohort_month AS VARCHAR) AS cohort_month,
                CAST(r.month_index AS INTEGER) AS month_index,
                (r.active_customers * 100.0 / c.total_customers) AS retention_pct
            FROM retention r
            JOIN cohort_sizes c ON r.cohort_month = c.cohort_month
            WHERE r.month_index <= 5 
            ORDER BY r.cohort_month DESC, r.month_index ASC
            LIMIT 30;
        """
        return self.conn.sql(query).pl()

    def run_f5_promotion_leakage(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                campaign_name,
                SUM(gross_sales) AS gross_revenue,
                SUM(discount_amount) AS total_discount_given,
                (SUM(discount_amount) / NULLIF(SUM(gross_sales), 0)) * 100 AS discount_leakage_pct
            FROM '{self.sales_csv}'
            WHERE campaign_name IS NOT NULL
            GROUP BY campaign_name
            ORDER BY discount_leakage_pct DESC
            LIMIT 10;
        """
        return self.conn.sql(query).pl()

    def run_f6_logistics_sla(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                warehouse,
                AVG(delivery_days) AS avg_actual_days,
                AVG(estimated_delivery_days) AS avg_estimated_days,
                AVG(delivery_days - estimated_delivery_days) AS avg_delivery_lag,
                COUNT(*) FILTER (WHERE delivery_status != 'On Time') AS late_deliveries
            FROM '{self.sales_csv}'
            GROUP BY warehouse
            ORDER BY late_deliveries DESC
            LIMIT 5;
        """
        return self.conn.sql(query).pl()

    def run_f7_return_root_cause(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                return_reason,
                COUNT(order_id) AS total_returns,
                SUM(net_sales) AS lost_revenue
            FROM '{self.sales_csv}'
            WHERE return_status IS NOT NULL AND return_reason IS NOT NULL
            GROUP BY return_reason
            ORDER BY total_returns DESC
            LIMIT 5;
        """
        return self.conn.sql(query).pl()

    def run_f8_multichannel_attribution(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                s.sales_channel,
                COUNT(s.order_id) AS total_orders,
                SUM(s.net_sales) AS net_revenue,
                SUM(c.customer_acquisition_cost) AS total_ad_spend,
                (SUM(s.net_sales) / NULLIF(SUM(c.customer_acquisition_cost), 0)) AS roas
            FROM '{self.sales_csv}' s
            LEFT JOIN '{self.customers_csv}' c ON s.customer_id = c.customer_id
            GROUP BY s.sales_channel
            ORDER BY roas DESC;
        """
        return self.conn.sql(query).pl()

    def run_f9_supplier_velocity(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                p.supplier,
                SUM(o.quantity) AS total_units_sold,
                SUM(o.profit) AS total_profit_generated,
                AVG(p.product_rating) AS avg_supplier_rating
            FROM '{self.items_csv}' AS o
            LEFT JOIN '{self.catalog_csv}' AS p ON o.product_id = p.product_id
            WHERE p.supplier IS NOT NULL
            GROUP BY p.supplier
            ORDER BY total_units_sold DESC
            LIMIT 5;
        """
        return self.conn.sql(query).pl()

    def run_f10_voc_sentiment(self) -> pl.DataFrame:
        query = f"""
            SELECT 
                review_sentiment,
                COUNT(order_id) AS review_count,
                AVG(customer_rating) AS avg_rating,
                AVG(profit_margin_percentage) AS avg_profit_margin
            FROM '{self.sales_csv}'
            WHERE review_sentiment IS NOT NULL
            GROUP BY review_sentiment
            ORDER BY avg_rating DESC;
        """
        return self.conn.sql(query).pl()