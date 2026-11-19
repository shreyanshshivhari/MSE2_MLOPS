"""
src/prepare_data.py
===================
PURPOSE (MLOps context):
  "Data preparation" is the first stage of any ML pipeline.
  This script turns raw Kaggle CSV dumps (or generates realistic samples)
  into clean .txt files inside data/raw/<class_name>/.
  Keeping data as plain-text files makes it trivial to version with DVC.

HOW TO RUN:
  python src/prepare_data.py                   # basic: 3 classes
  python src/prepare_data.py --add-bank-statements  # add 4th class (drift demo)
  python src/prepare_data.py --n 60            # only 60 docs per class
"""

import argparse
import csv
import os
import random
import textwrap
from pathlib import Path

# ── Project paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root
RAW = ROOT / "data" / "raw"                 # where clean .txt files live
EXTERNAL = ROOT / "data" / "external"       # where the user drops Kaggle CSVs
SAMPLE_UPLOADS = ROOT / "sample_uploads"    # demo files for the API

# ── Seed for reproducibility ────────────────────────────────────────────────────
random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: Synthetic text generators
#  These produce realistic-looking documents so the pipeline works immediately,
#  even before the user downloads any Kaggle dataset.
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_invoice(idx: int) -> str:
    """Return a realistic synthetic invoice as a plain-text string."""
    companies = [
        "Apex Solutions Ltd", "BlueSky Consulting", "NextGen Technologies",
        "Orion Supplies Co", "Pinnacle Services", "Vertex Corp",
        "Summit Enterprises", "Global Trade Inc", "Prime Logistics",
        "Alpha Systems",
    ]
    items = [
        ("Software License", 1, 4500),
        ("Annual Support Contract", 1, 1200),
        ("Hardware Components", random.randint(2, 10), random.randint(200, 800)),
        ("Cloud Hosting (monthly)", 1, random.randint(300, 2000)),
        ("Consulting Services", random.randint(5, 40), 150),
        ("Office Supplies", random.randint(10, 50), random.randint(10, 80)),
        ("Training Workshop", 1, random.randint(500, 3000)),
        ("Maintenance Fee", 1, random.randint(400, 1500)),
    ]
    company = random.choice(companies)
    inv_no = f"INV-{2024 * 100 + idx:06d}"
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    item_name, qty, unit_price = random.choice(items)
    total = qty * unit_price
    tax = round(total * 0.18, 2)

    return textwrap.dedent(f"""
        INVOICE
        ───────────────────────────────────────
        Invoice Number : {inv_no}
        Invoice Date   : {day:02d}/{month:02d}/2024
        Due Date       : {day:02d}/{(month % 12) + 1:02d}/2024

        Bill To:
          {company}
          123 Business Park, Suite {random.randint(10, 99)}
          New Delhi – 110001

        Bill From:
          MyCompany Pvt. Ltd.
          456 Corporate Hub, Floor {random.randint(1, 20)}
          Mumbai – 400001

        ───────────────────────────────────────
        ITEM DESCRIPTION        QTY    UNIT PRICE    AMOUNT
        {item_name:<24} {qty:<6} {unit_price:<13,.2f} {total:,.2f}
        ───────────────────────────────────────
        Sub Total                                    {total:,.2f}
        GST (18%)                                    {tax:,.2f}
        TOTAL DUE                                    {total + tax:,.2f}
        ───────────────────────────────────────
        Payment Terms : Net 30 days
        Bank Details  : HDFC Bank, A/C 00123456789, IFSC HDFC0001234

        Thank you for your business!
        For queries: billing@mycompany.com | +91-98765-43210
    """).strip()


def _rand_resume(idx: int) -> str:
    """Return a realistic synthetic resume as a plain-text string."""
    first_names = ["Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Anjali",
                   "Rohit", "Nisha", "Arjun", "Divya", "Karthik", "Meera"]
    last_names = ["Sharma", "Patel", "Verma", "Singh", "Kumar", "Gupta",
                  "Joshi", "Nair", "Reddy", "Iyer", "Mehta", "Das"]
    roles = ["Software Engineer", "Data Analyst", "Product Manager",
             "DevOps Engineer", "ML Engineer", "Backend Developer",
             "Frontend Developer", "Full Stack Developer", "QA Engineer"]
    skills_pool = [
        "Python", "Java", "SQL", "Docker", "Kubernetes", "React", "Node.js",
        "TensorFlow", "PyTorch", "scikit-learn", "AWS", "GCP", "Azure",
        "FastAPI", "Django", "PostgreSQL", "MongoDB", "Redis", "Git", "Linux",
        "Tableau", "Power BI", "Spark", "Kafka", "Airflow",
    ]
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    role = random.choice(roles)
    skills = random.sample(skills_pool, k=random.randint(6, 10))
    exp_yrs = random.randint(1, 8)

    return textwrap.dedent(f"""
        {name.upper()}
        {role}
        Email: {name.lower().replace(' ', '.')}@email.com | Phone: +91-9{random.randint(100000000, 999999999)}
        LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}{idx}
        Location: Bangalore, India

        ── PROFESSIONAL SUMMARY ────────────────────────────────────────────────
        Results-driven {role} with {exp_yrs}+ years of experience building
        scalable software solutions. Passionate about clean code, automation,
        and continuous learning. Proven track record of delivering projects
        on time in Agile environments.

        ── SKILLS ──────────────────────────────────────────────────────────────
        {", ".join(skills)}

        ── WORK EXPERIENCE ─────────────────────────────────────────────────────
        {role} | TechCorp India Pvt. Ltd.           2022 – Present
          • Led development of microservices handling 1M+ daily requests.
          • Reduced deployment time by 40% by implementing CI/CD pipelines.
          • Mentored 3 junior developers; conducted weekly code reviews.

        Junior {role} | StartupXYZ                  2020 – 2022
          • Built RESTful APIs consumed by mobile and web clients.
          • Wrote unit and integration tests achieving 85%+ coverage.
          • Collaborated with cross-functional teams in 2-week sprints.

        ── EDUCATION ───────────────────────────────────────────────────────────
        B.Tech in Computer Science | IIT Bombay            2016 – 2020
        CGPA: {random.uniform(7.5, 9.5):.2f} / 10

        ── CERTIFICATIONS ──────────────────────────────────────────────────────
        AWS Certified Solutions Architect – Associate (2023)
        Google Professional Data Engineer (2022)

        ── PROJECTS ────────────────────────────────────────────────────────────
        Smart Inventory System: Reduced stock-outs by 30% using ML forecasting.
        Real-time Chat App: WebSocket-based chat with {random.randint(500,5000)} concurrent users.
    """).strip()


def _rand_certificate(idx: int) -> str:
    """Return a realistic synthetic certificate as a plain-text string."""
    courses = [
        "Machine Learning with Python", "Cloud Computing Fundamentals",
        "Advanced Data Structures", "Deep Learning Specialisation",
        "DevOps and CI/CD Pipelines", "Cybersecurity Essentials",
        "Python for Data Science", "Docker and Kubernetes Mastery",
        "Natural Language Processing", "Full Stack Web Development",
    ]
    issuers = [
        "Coursera", "edX", "Udemy", "NPTEL", "Google Developer Training",
        "Microsoft Learn", "AWS Training & Certification", "LinkedIn Learning",
    ]
    names = ["Aarav Sharma", "Priya Patel", "Rahul Verma", "Sneha Singh",
             "Vikram Kumar", "Anjali Gupta", "Rohit Joshi", "Nisha Nair"]
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    course = random.choice(courses)
    issuer = random.choice(issuers)
    name = random.choice(names)
    cert_id = f"CERT-{issuer[:3].upper()}-{2024 * 10 + idx:08d}"

    return textwrap.dedent(f"""
        CERTIFICATE OF COMPLETION
        ═══════════════════════════════════════════════════════════
        This is to certify that

                        {name}

        has successfully completed the course

                "{course}"

        offered by {issuer}.

        Duration   : {random.randint(10, 80)} hours
        Score      : {random.randint(75, 100)}%
        Issue Date : {day:02d}/{month:02d}/2024
        Certificate ID : {cert_id}

        ───────────────────────────────────────────────────────────
        {issuer} | Authorised Signatory
        verify.{issuer.lower().replace(' ', '')}.com/certificate/{cert_id}
        ═══════════════════════════════════════════════════════════

        This certificate confirms the holder's proficiency in the
        subject matter as assessed through quizzes, assignments,
        and a final examination administered online.
    """).strip()


def _rand_bank_statement(idx: int) -> str:
    """Return a realistic synthetic bank statement as a plain-text string."""
    banks = ["HDFC Bank", "ICICI Bank", "State Bank of India",
             "Axis Bank", "Kotak Mahindra Bank", "Punjab National Bank"]
    names = ["Aarav Sharma", "Priya Patel", "Rahul Verma", "Sneha Singh"]
    bank = random.choice(banks)
    name = random.choice(names)
    acc_no = f"XXXX XXXX {random.randint(1000, 9999)}"
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month = random.choice(month_names)
    opening_bal = random.uniform(10000, 100000)

    # Generate 8-12 random transactions
    txns = []
    bal = opening_bal
    for t in range(random.randint(8, 12)):
        day = random.randint(1, 28)
        is_credit = random.random() > 0.4
        amount = round(random.uniform(500, 25000), 2)
        bal = bal + amount if is_credit else bal - amount
        bal = max(bal, 0)
        cr_dr = "CR" if is_credit else "DR"
        desc = random.choice([
            "NEFT Transfer", "UPI Payment", "ATM Withdrawal",
            "Online Purchase", "Salary Credit", "Bill Payment",
            "EMI Debit", "Interest Credit", "Refund Credit", "Insurance Premium",
        ])
        txns.append(f"  {day:02d} {month[:3]} 2024   {desc:<25} {amount:>10,.2f} {cr_dr}   {bal:>12,.2f}")

    txns_str = "\n".join(txns)

    return textwrap.dedent(f"""
        {bank.upper()}
        ACCOUNT STATEMENT
        ══════════════════════════════════════════════════════════════
        Account Holder : {name}
        Account Number : {acc_no}
        Account Type   : Savings Account
        Statement Period: 01 {month} 2024 – 30 {month} 2024
        Branch         : MG Road, Bangalore – 560001
        IFSC Code      : {bank[:4].upper().replace(' ', '')[:4]}0001234

        ──────────────────────────────────────────────────────────────
        Opening Balance : ₹{opening_bal:,.2f}

        DATE          DESCRIPTION               AMOUNT       CR/DR   BALANCE
        ──────────────────────────────────────────────────────────────
{txns_str}
        ──────────────────────────────────────────────────────────────
        Closing Balance : ₹{bal:,.2f}

        Generated for: {name}
        Customer Care : 1800-XXX-XXXX | {bank.lower().replace(' ', '')}.com
        ══════════════════════════════════════════════════════════════
        This is a computer-generated statement and requires no signature.
    """).strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: Kaggle CSV loaders
#  If the user has placed a CSV from Kaggle in data/external/, we read real
#  document text from it instead of generating synthetic samples.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_csv_texts(csv_path: Path, text_columns: list[str], n: int) -> list[str]:
    """
    Read up to `n` non-empty text strings from the first matching column
    in `text_columns` from the CSV at `csv_path`.
    Returns an empty list if the file doesn't exist or no column matches.
    """
    if not csv_path.exists():
        return []

    texts = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        # Find which column holds the text
        col = next((c for c in text_columns if c in (reader.fieldnames or [])), None)
        if col is None:
            print(f"  [WARN] None of {text_columns} found in {csv_path.name}. "
                  f"Available columns: {reader.fieldnames}")
            return []
        for row in reader:
            text = row[col].strip()
            if text:
                texts.append(text)
            if len(texts) >= n:
                break

    print(f"  [OK] Loaded {len(texts)} rows from {csv_path.name} (column='{col}')")
    return texts


def _find_external_csv(pattern_starts: list[str]) -> Path | None:
    """
    Look inside data/external/ for the first CSV file whose name starts
    with any of the given prefixes (case-insensitive).
    """
    if not EXTERNAL.exists():
        return None
    for f in EXTERNAL.iterdir():
        if f.suffix.lower() == ".csv":
            name_lower = f.name.lower()
            if any(name_lower.startswith(p.lower()) for p in pattern_starts):
                return f
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: File writers
#  Save each document as an individual .txt file so sklearn's load_files()
#  can read them (it treats each file as one document, the folder as the label).
# ═══════════════════════════════════════════════════════════════════════════════

def _write_texts(folder: Path, texts: list[str], label: str) -> None:
    """Write each text string to its own numbered .txt file inside `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        (folder / f"{label}_{i:04d}.txt").write_text(text, encoding="utf-8")
    print(f"  → Wrote {len(texts)} files to {folder.relative_to(ROOT)}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Per-class preparation functions
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_invoices(n: int) -> None:
    """
    Try to load invoice text from a Kaggle CSV first.
    The Invoice NER Dataset (nikitpatel/invoice-ner-dataset) has an 'Input' column.
    Fall back to synthetic generation if the CSV is not present.
    """
    print("\n[1/4] Preparing INVOICES ...")
    folder = RAW / "invoices"

    # Look for the Kaggle invoice CSV in data/external/
    csv_path = _find_external_csv(["invoice"])
    texts = _load_csv_texts(csv_path, ["Input", "text", "content"], n) if csv_path else []

    if texts:
        texts = random.sample(texts, min(len(texts), n))  # cap at n
    else:
        # No CSV found → synthesise
        print("  [INFO] No Kaggle invoice CSV found in data/external/. "
              "Generating synthetic samples.")
        print("  [HINT] Download nikitpatel/invoice-ner-dataset from Kaggle,")
        print("         place the CSV in data/external/, and re-run this script.")
        texts = [_rand_invoice(i) for i in range(n)]

    _write_texts(folder, texts, "invoice")


def prepare_resumes(n: int) -> None:
    """
    Try to load resume text from a Kaggle CSV first.
    The Resume Dataset (snehaanbhawal/resume-dataset) has a 'Resume_str' column.
    Fall back to synthetic generation if the CSV is not present.
    """
    print("\n[2/4] Preparing RESUMES ...")
    folder = RAW / "resumes"

    csv_path = _find_external_csv(["resume", "Resume"])
    # Try multiple possible column names from different Kaggle resume datasets
    texts = _load_csv_texts(csv_path, ["Resume_str", "resume", "text", "content", "Resume"], n) if csv_path else []

    if texts:
        texts = random.sample(texts, min(len(texts), n))
    else:
        print("  [INFO] No Kaggle resume CSV found in data/external/. "
              "Generating synthetic samples.")
        print("  [HINT] Download snehaanbhawal/resume-dataset from Kaggle,")
        print("         place the CSV in data/external/, and re-run this script.")
        texts = [_rand_resume(i) for i in range(n)]

    _write_texts(folder, texts, "resume")


def prepare_certificates(n: int) -> None:
    """
    Certificates are always generated synthetically.
    You can also drop any certificate CSV into data/external/ with a
    'text' or 'content' column and name it starting with 'cert'.
    """
    print("\n[3/4] Preparing CERTIFICATES ...")
    folder = RAW / "certificates"

    csv_path = _find_external_csv(["cert"])
    texts = _load_csv_texts(csv_path, ["text", "content", "certificate"], n) if csv_path else []

    if not texts:
        texts = [_rand_certificate(i) for i in range(n)]

    _write_texts(folder, texts, "certificate")


def prepare_bank_statements(n: int) -> None:
    """
    Bank statements are the NEW class added in v2 to demonstrate
    model retraining with an expanded dataset — a core MLOps concept.
    """
    print("\n[4/4] Preparing BANK STATEMENTS (new class for v2 retraining) ...")
    folder = RAW / "bank_statements"

    csv_path = _find_external_csv(["bank", "statement"])
    texts = _load_csv_texts(csv_path, ["text", "content", "statement"], n) if csv_path else []

    if not texts:
        texts = [_rand_bank_statement(i) for i in range(n)]

    _write_texts(folder, texts, "bank_statement")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Sample upload files
#  Create two ready-to-use test files for demoing the /predict API endpoint.
# ═══════════════════════════════════════════════════════════════════════════════

def create_sample_uploads() -> None:
    """
    Write two demo files into sample_uploads/ so anyone can immediately
    test the API without preparing their own documents.
    """
    SAMPLE_UPLOADS.mkdir(parents=True, exist_ok=True)

    # A clean test invoice (the model should predict 'invoices' with high confidence)
    invoice_text = _rand_invoice(9999)
    (SAMPLE_UPLOADS / "test_invoice.txt").write_text(invoice_text, encoding="utf-8")

    # A bank statement that the v1 model has never seen – should trigger drift alert
    bank_text = _rand_bank_statement(9999)
    (SAMPLE_UPLOADS / "test_unknown_bank_statement.txt").write_text(bank_text, encoding="utf-8")

    print("\n[SAMPLE_UPLOADS] Created:")
    print("  sample_uploads/test_invoice.txt")
    print("  sample_uploads/test_unknown_bank_statement.txt")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data/raw/ for the Smart Document Classifier."
    )
    parser.add_argument(
        "--n", type=int, default=120,
        help="Max documents per class (default: 120)."
    )
    parser.add_argument(
        "--add-bank-statements", action="store_true",
        help="Also create data/raw/bank_statements/ (v2 retraining demo)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Smart Document Classification – Data Preparation")
    print("=" * 60)
    print(f"  Target docs per class : {args.n}")
    print(f"  External data folder  : {EXTERNAL.relative_to(ROOT)}/")

    prepare_invoices(args.n)
    prepare_resumes(args.n)
    prepare_certificates(args.n)

    if args.add_bank_statements:
        prepare_bank_statements(args.n)

    create_sample_uploads()

    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print("  Next step: python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
