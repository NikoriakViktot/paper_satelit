import requests
import logging
import time
import csv
from pathlib import Path
from lxml import etree
from tqdm import tqdm

# ================= CONFIG =================

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

BASE = Path("/home/viktornikoriak/paper_satelit/data/literature")
PDF_DIR = BASE / "pdf"
XML_DIR = BASE / "grobid_xml"

XML_DIR.mkdir(exist_ok=True)

CSV_LOG = "grobid_results.csv"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# ================= LOGGING =================

logging.basicConfig(
    filename="grobid_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= CSV =================

def log_csv(file_name, status, reason=""):
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([file_name, status, reason])

# ================= VALIDATION =================

def validate_xml(xml_text):
    try:
        root = etree.fromstring(xml_text.encode())

        has_abstract = bool(root.xpath("//tei:abstract", namespaces=NS))
        has_body = bool(root.xpath("//tei:body", namespaces=NS))
        has_title = bool(root.xpath("//tei:titleStmt/tei:title", namespaces=NS))

        return {
            "valid_xml": True,
            "has_abstract": has_abstract,
            "has_body": has_body,
            "has_title": has_title
        }

    except Exception:
        return {
            "valid_xml": False,
            "has_abstract": False,
            "has_body": False,
            "has_title": False
        }


# ================= CORE =================

def process_pdf(pdf_path: Path, max_retries=3):
    out_file = XML_DIR / f"{pdf_path.stem}.tei.xml"

    if out_file.exists():
        logging.info(f"{pdf_path.name} | SKIPPED_EXISTS")
        log_csv(pdf_path.name, "SKIPPED", "EXISTS")
        print(f"⏩ SKIP: {pdf_path.name}")
        return "SKIPPED"

    print(f"📄 START: {pdf_path.name}")

    for attempt in range(max_retries):
        try:
            with open(pdf_path, "rb") as f:
                response = requests.post(
                    GROBID_URL,
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "0",
                        "teiCoordinates": "1"
                    },
                    timeout=120
                )

            if response.status_code != 200:
                reason = f"HTTP_{response.status_code}"
                logging.error(f"{pdf_path.name} | FAIL | {reason}")
                log_csv(pdf_path.name, "FAIL", reason)
                print(f"❌ FAIL: {pdf_path.name} ({reason})")
                continue

            if not response.text.strip():
                reason = "EMPTY_XML"
                logging.error(f"{pdf_path.name} | FAIL | {reason}")
                log_csv(pdf_path.name, "FAIL", reason)
                print(f"❌ EMPTY: {pdf_path.name}")
                continue

            validation = validate_xml(response.text)

            if not validation["valid_xml"]:
                logging.error(f"{pdf_path.name} | INVALID_XML")
                log_csv(pdf_path.name, "FAIL", "INVALID_XML")
                print(f"❌ INVALID XML: {pdf_path.name}")
                return "FAIL"

            flags = []

            if not validation["has_body"]:
                flags.append("NO_BODY")

            if not validation["has_abstract"]:
                flags.append("NO_ABSTRACT")

            if not validation["has_title"]:
                flags.append("NO_TITLE")

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(response.text)

            if flags:
                logging.warning(f"{pdf_path.name} | PARTIAL | {'|'.join(flags)}")
                log_csv(pdf_path.name, "PARTIAL", "|".join(flags))
                print(f"⚠️ PARTIAL: {pdf_path.name} ({'|'.join(flags)})")
                return "PARTIAL"
            else:
                logging.info(f"{pdf_path.name} | SUCCESS | FULL")
                log_csv(pdf_path.name, "SUCCESS", "FULL")
                print(f"✅ SUCCESS: {pdf_path.name}")
                return "SUCCESS"

        except requests.exceptions.Timeout:
            logging.error(f"{pdf_path.name} | TIMEOUT")
            log_csv(pdf_path.name, "FAIL", "TIMEOUT")
            print(f"⏱ TIMEOUT: {pdf_path.name}")

        except requests.exceptions.ConnectionError:
            logging.error(f"{pdf_path.name} | CONNECTION_ERROR")
            log_csv(pdf_path.name, "FAIL", "CONNECTION_ERROR")
            print(f"🔌 CONNECTION ERROR: {pdf_path.name}")

        except Exception as e:
            logging.error(f"{pdf_path.name} | UNKNOWN_ERROR | {str(e)}")
            log_csv(pdf_path.name, "FAIL", f"UNKNOWN_{str(e)}")
            print(f"💥 ERROR: {pdf_path.name} ({str(e)})")

        time.sleep(2)

    logging.error(f"{pdf_path.name} | FAILED_AFTER_RETRIES")
    log_csv(pdf_path.name, "FAIL", "FAILED_AFTER_RETRIES")
    print(f"❌ FINAL FAIL: {pdf_path.name}")
    return "FAIL"

# ================= RUN =================

def run():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    logging.info(f"PIPELINE_START | {len(pdfs)} files")

    stats = {
        "SUCCESS": 0,
        "PARTIAL": 0,
        "FAIL": 0,
        "SKIPPED": 0
    }

    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        result = process_pdf(pdf)
        if result:
            stats[result] += 1

    logging.info("PIPELINE_DONE")

    print("\n📊 FINAL STATS:")
    for k, v in stats.items():
        print(f"{k}: {v}")

# ================= ENTRY =================

if __name__ == "__main__":
    run()