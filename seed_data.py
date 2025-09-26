# seed_data.py — generate large mock datasets for the Consumer HR GenAI app
import os, csv, random, string
from datetime import date, datetime, timedelta

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)

N_EMPLOYEES   = 150
N_ORDERS      = 150
N_PRODUCTS    = 200
DAYS_OF_SHIFTS = 7

first_names = ["Aarav","Vivaan","Aditya","Vihaan","Arjun","Reyansh","Muhammad","Sai","Arnav","Ayaan","Isha","Aanya","Ananya","Diya","Myra","Sara","Ira","Aadhya","Siya","Zara","Meera","Riya","Anika","Avni","Pari","Kavya","Aarohi","Navya","Nitya","Mahika"]
last_names  = ["Sharma","Verma","Iyer","Menon","Agarwal","Khan","Patel","Nair","Reddy","Rao","Gupta","Kapoor","Bose","Mitra","Naidu","Mehta","Joshi","Kulkarni","Desai","Chopra","Sinha","Pandey","Tripathi","Yadav","Jadeja","Das","Paul","Lal","Chauhan","Shetty"]
roles       = ["Store Associate","Store Cashier","Beauty Advisor","Warehouse Picker","Warehouse Packer","Inventory Exec","Store Manager","Assistant Manager","HR Associate","Trainer"]
managers    = ["Priya Nair","Rahul Khanna","Neeraj Gupta","Ankit Rao","Zoya Khan","Ritu Mehta","Sanjay Patel","Kunal Kapoor"]
cities_stores = {
    "Mumbai": ["Bandra","Andheri","Powai","Vashi"],
    "Delhi": ["Saket","Rohini","CP","Dwarka"],
    "Bengaluru": ["Indiranagar","Whitefield","Koramangala","HSR"],
    "Pune": ["Koregaon Park","Baner","Hinjawadi","Kothrud"],
    "Hyderabad": ["Banjara Hills","Gachibowli","Kukatpally"],
    "Chennai": ["T Nagar","Velachery","Adyar"],
    "Kolkata": ["Salt Lake","Park Street","Howrah"],
    "Ahmedabad": ["SG Highway","Maninagar"],
    "Jaipur": ["MI Road","Malviya Nagar"]
}
shifts_list = ["9:00-18:00","13:00-22:00"]
carriers    = ["BlueDart","Delhivery","DTDC","Ecom Express","Shadowfax","XpressBees","India Post"]
order_items = ["Wireless Mouse","Bluetooth Headphones","Face Serum 50ml","Shampoo 200ml","Conditioner 200ml","Beard Oil 30ml","Sunscreen SPF50","Body Lotion 250ml","Lip Balm","Hand Sanitizer 100ml","Kajal","Liquid Lipstick","Face Wash 100ml","Hair Wax","Deodorant"]
product_pool = [
    ("SKU", "Wireless Mouse"),("SKU","Bluetooth Headphones"),("SKU","Face Serum 50ml"),
    ("SKU","Shampoo 200ml"),("SKU","Conditioner 200ml"),("SKU","Sunscreen SPF50"),
    ("SKU","Body Lotion 250ml"),("SKU","Lip Balm"),("SKU","Hand Sanitizer 100ml"),
    ("SKU","Kajal"),("SKU","Liquid Lipstick"),("SKU","Face Wash 100ml"),
    ("SKU","Hair Wax"),("SKU","Deodorant")
]

def rand_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def rand_city_store():
    city = random.choice(list(cities_stores.keys()))
    store = random.choice(cities_stores[city])
    return city, store

def rand_join_date():
    start = datetime(2021,1,1)
    days = (datetime.now() - start).days
    d = start + timedelta(days=random.randint(0, max(days,1)))
    return d.date().isoformat()

def rand_phone():
    return "9" + "".join(random.choices(string.digits, k=9))

today = date.today()
today_str = today.isoformat()

# Policies
policies = [
    ["policy_id","topic","content"],
    ["P001","Leave Policy","Employees accrue 1.5 days of paid leave per month. Carry forward up to 12 days. Blackout dates: Diwali week for store roles unless pre-approved."],
    ["P002","Attendance & Shifts","Store shifts: 9am-6pm and 1pm-10pm. 30-min break included. Late >10 mins counted as half-day after 3 occurrences/month."],
    ["P003","Overtime","OT payable at 1.5x hourly rate for >9 hours/day or >48 hours/week. Prior manager approval required."],
    ["P004","Uniform & Grooming","Uniform mandatory on floor. Shoes must be closed-toe. Name badge visible at all times."],
    ["P005","Training & L&D","Mandatory onboarding modules in first 14 days. Quarterly product knowledge assessment for store staff."],
    ["P006","Code of Conduct","Zero tolerance for harassment or customer misrepresentation. Violations lead to disciplinary action including termination."]
]
with open(os.path.join(DATA_DIR,"hr_policies.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(policies)

# FAQs (base + extras)
faqs = [["question","answer"],
    ["How many paid leaves do I get per month?","You accrue 1.5 days of paid leave per month; you can carry forward up to 12 days."],
    ["What are the standard store shift timings?","Two shifts: 9am–6pm and 1pm–10pm with a 30-minute break."],
    ["Is overtime paid and at what rate?","Yes. OT is 1.5× hourly rate if you work >9 hours/day or >48 hours/week, with prior approval."],
    ["Can I take leave during Diwali week?","For store roles, Diwali week is a blackout period unless you have prior approval from your manager."],
    ["What is the late policy?","Arrivals >10 min late count as half-day after 3 occurrences in a month."],
    ["Do I have to wear a uniform?","Yes. Uniform and closed-toe shoes are mandatory; name badge must be visible."],
    ["Are there mandatory trainings?","Yes. Complete onboarding in 14 days and quarterly product knowledge assessments."]
]
for i in range(50):
    faqs.append([f"How do I mark attendance if biometric is down #{i+1}?","Use the manual register with manager sign-off. HR will reconcile at EOD."])
with open(os.path.join(DATA_DIR,"hr_faq.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(faqs)

# Employees
employees = [["emp_id","name","role","location","manager","join_date","leave_balance","phone"]]
for i in range(1, N_EMPLOYEES+1):
    emp_id = f"C{i:03d}"
    name = rand_name()
    role = random.choice(roles)
    city, store = rand_city_store()
    loc  = f"{city} - {store}"
    mgr  = random.choice(managers)
    join = rand_join_date()
    lb   = random.randint(0, 20)
    phone = rand_phone()
    employees.append([emp_id, name, role, loc, mgr, join, lb, phone])
with open(os.path.join(DATA_DIR,"employees.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(employees)

# Shifts (today + next days; ~80% scheduled)
shifts = [["emp_id","date","shift","store"]]
for day in range(DAYS_OF_SHIFTS):
    d = (today + timedelta(days=day)).isoformat()
    for i in range(1, N_EMPLOYEES+1):
        if random.random() < 0.8:
            _, s = rand_city_store()
            shifts.append([f"C{i:03d}", d, random.choice(shifts_list), s])
with open(os.path.join(DATA_DIR,"shifts.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(shifts)

# Training (1–3 modules per employee)
training_modules = ["Onboarding Basics","POS Compliance","Product Knowledge","Safety & Handling","Customer Experience","Loss Prevention"]
training_status   = ["Completed","Pending","In Progress"]
training = [["emp_id","module","status","due_date"]]
for i in range(1, N_EMPLOYEES+1):
    n = random.randint(1,3)
    picks = random.sample(training_modules, k=n)
    for m in picks:
        delta = random.randint(-60, 60)
        due = (today + timedelta(days=delta)).isoformat()
        training.append([f"C{i:03d}", m, random.choice(training_status), due])
with open(os.path.join(DATA_DIR,"training.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(training)

# Orders
orders = [["order_id","customer","item","status","eta","carrier","tracking"]]
order_statuses = ["Processing","Shipped","Out for Delivery","Delivered","Cancelled"]
for i in range(1, N_ORDERS+1):
    oid = f"O-{today.year}-{1000+i}"
    customer = rand_name()
    item = random.choice(order_items)
    status = random.choices(order_statuses, weights=[30,25,20,20,5])[0]
    eta_days = random.randint(0, 6)
    eta = (today + timedelta(days=eta_days)).isoformat()
    carrier = random.choice(carriers) if status in ["Shipped","Out for Delivery","Delivered"] else ""
    tracking = "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) if carrier else ""
    orders.append([oid, customer, item, status, eta, carrier, tracking])
with open(os.path.join(DATA_DIR,"orders.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(orders)

# Products
products = [["sku","name","city","store","stock"]]
for i in range(1, N_PRODUCTS+1):
    name = random.choice(product_pool)[1]
    city, store = rand_city_store()
    stock = max(0, int(random.gauss(8, 6)))  # many small stocks, some zeros
    products.append([f"SKU-{i:04d}", name, city, store, stock])
with open(os.path.join(DATA_DIR,"products.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(products)

# Tickets (empty headers)
with open(os.path.join(DATA_DIR,"tickets.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["ticket_id","created_at","channel","user_input","intent","confidence","status","notes"])

print("✅ Seeded large datasets in ./data")
