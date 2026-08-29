import sqlite3
import random
from pathlib import Path

random.seed(42)

DB_PATH = Path(__file__).resolve().parent / "sales.db"
if DB_PATH.exists():
    try:
        DB_PATH.unlink()
    except PermissionError:
        # File is locked; write to a temp path then atomic-replace at end.
        import time
        DB_PATH = DB_PATH.with_name(f"sales_{int(time.time())}.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript(
    """
    CREATE TABLE regions (
        region_id   INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        country     TEXT NOT NULL,
        manager     TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        first_name  TEXT NOT NULL,
        last_name   TEXT NOT NULL,
        email       TEXT NOT NULL UNIQUE,
        phone       TEXT,
        region_id   INTEGER,
        signup_date TEXT NOT NULL,
        FOREIGN KEY (region_id) REFERENCES regions(region_id)
    );

    CREATE TABLE products (
        product_id  INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        category    TEXT NOT NULL,
        unit_price  REAL NOT NULL,
        stock_qty   INTEGER NOT NULL,
        launch_date TEXT NOT NULL
    );

    CREATE TABLE orders (
        order_id    INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        product_id  INTEGER NOT NULL,
        quantity    INTEGER NOT NULL,
        total_price REAL NOT NULL,
        order_date  TEXT NOT NULL,
        status      TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """
)

region_names = [
    ("North", "USA"), ("South", "USA"), ("East", "USA"), ("West", "USA"),
    ("Ontario", "Canada"), ("Quebec", "Canada"), ("British Columbia", "Canada"),
    ("London", "UK"), ("Manchester", "UK"), ("Bavaria", "Germany"),
    ("Berlin", "Germany"), ("Ile-de-France", "France"), ("Madrid", "Spain"),
    ("Catalonia", "Spain"), ("Lazio", "Italy"), ("Lombardy", "Italy"),
    ("Maharashtra", "India"), ("Karnataka", "India"), ("New South Wales", "Australia"),
    ("Victoria", "Australia"),
]

def make_region_name(i):
    base = i % len(region_names)
    name, country = region_names[base]
    cycle = i // len(region_names)
    if cycle > 0:
        suffix = ["Central", "Metro", "Coastal", "Highlands", "Valley", "Plateau", "Delta", "Riverside", "Harbor", "Summit"][cycle - 1]
        name = f"{suffix} {name}"
    return name, country

managers = [
    "Alice Johnson", "Bob Smith", "Carlos Diaz", "Dana Lee", "Erik Hanson",
    "Fatima Khan", "George Brown", "Hiro Tanaka", "Ivan Petrov", "Julia Costa"
]

for i, (name, country) in enumerate([make_region_name(i) for i in range(100)], start=1):
    cur.execute(
        "INSERT INTO regions (region_id, name, country, manager, created_at) VALUES (?, ?, ?, ?, ?)",
        (i, name, country, managers[i % len(managers)], f"2020-0{random.randint(1,9)}-1{random.randint(0,9)}"),
    )

first_names = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Elizabeth",
               "David","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen",
               "Christopher","Nancy","Daniel","Lisa","Matthew","Margaret","Anthony","Sandra","Mark","Ashley",
               "Donald","Kimberly","Steven","Emily","Paul","Donna","Andrew","Michelle","Joshua","Carol",
               "Kenneth","Amanda","Kevin","Melissa","Brian","Deborah","George","Stephanie","Edward","Rebecca"]
last_names = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
              "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
              "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
              "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
              "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts"]

for i in range(1, 101):
    fn, ln = random.choice(first_names), random.choice(last_names)
    email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
    phone = f"+1-{random.randint(200,989)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    signup = f"202{random.randint(0,4)}-0{random.randint(1,9)}-0{random.randint(1,9)}"
    cur.execute(
        "INSERT INTO customers (customer_id, first_name, last_name, email, phone, region_id, signup_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (i, fn, ln, email, phone, random.randint(1, 100), signup),
    )

product_catalog = [
    ("Wireless Mouse", "Electronics", 25.99),
    ("USB-C Cable", "Electronics", 12.49),
    ("Bluetooth Headphones", "Electronics", 79.99),
    ("4K Monitor", "Electronics", 329.00),
    ("Mechanical Keyboard", "Electronics", 119.50),
    ("Webcam HD", "Electronics", 54.30),
    ("Laptop Stand", "Office", 39.99),
    ("Notebook", "Office", 4.99),
    ("Pen Set", "Office", 9.75),
    ("Desk Lamp", "Office", 24.00),
    ("Office Chair", "Office", 189.00),
    ("Stapler", "Office", 8.50),
    ("Coffee Maker", "Kitchen", 89.99),
    ("Electric Kettle", "Kitchen", 34.95),
    ("Toaster", "Kitchen", 29.50),
    ("Blender", "Kitchen", 64.99),
    ("Cookware Set", "Kitchen", 149.00),
    ("Water Bottle", "Kitchen", 14.99),
    ("Running Shoes", "Apparel", 99.99),
    ("Cotton T-Shirt", "Apparel", 15.00),
    ("Denim Jeans", "Apparel", 49.99),
    ("Winter Jacket", "Apparel", 129.00),
    ("Sun Hat", "Apparel", 19.25),
    ("Sports Cap", "Apparel", 11.99),
    ("Yoga Mat", "Fitness", 28.00),
    ("Dumbbell Set", "Fitness", 75.50),
    ("Resistance Bands", "Fitness", 16.99),
    ("Treadmill", "Fitness", 599.00),
    ("Jump Rope", "Fitness", 7.50),
    ("Water Bottle Pro", "Fitness", 22.00),
]

for i in range(1, 101):
    name, category, price = product_catalog[(i - 1) % len(product_catalog)]
    name = f"{name} v{(i // len(product_catalog)) + 1}" if i > len(product_catalog) else name
    stock = random.randint(0, 500)
    launch = f"202{random.randint(0,4)}-0{random.randint(1,9)}-0{random.randint(1,9)}"
    cur.execute(
        "INSERT INTO products (product_id, name, category, unit_price, stock_qty, launch_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (i, name, category, float(price + random.uniform(-3, 3)), stock, launch),
    )

statuses = ["Pending", "Shipped", "Delivered", "Cancelled", "Refunded"]

for i in range(1, 101):
    cust_id = random.randint(1, 100)
    prod_id = random.randint(1, 100)
    qty = random.randint(1, 10)
    cur.execute("SELECT unit_price FROM products WHERE product_id = ?", (prod_id,))
    unit_price = cur.fetchone()[0]
    total = round(unit_price * qty, 2)
    order_date = f"202{random.randint(0,4)}-0{random.randint(1,9)}-0{random.randint(1,9)}"
    status = random.choice(statuses)
    cur.execute(
        "INSERT INTO orders (order_id, customer_id, product_id, quantity, total_price, order_date, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (i, cust_id, prod_id, qty, total, order_date, status),
    )

conn.commit()

cur.execute("SELECT name, (SELECT COUNT(*) FROM regions) AS n FROM (SELECT 'regions' AS name) UNION ALL "
            "SELECT 'customers', COUNT(*) FROM customers UNION ALL "
            "SELECT 'products', COUNT(*) FROM products UNION ALL "
            "SELECT 'orders', COUNT(*) FROM orders")
for name, n in cur.fetchall():
    print(f"{name}: {n} rows")

conn.close()
print(f"DB created at: {DB_PATH}")
