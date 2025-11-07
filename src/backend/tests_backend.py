"""
Comprehensive test suite for backend API
Run with: python src/backend/test_backend.py
"""
import requests
import json
from datetime import datetime
from typing import Dict, Optional

BASE_URL = "http://localhost:8000"

class Colors:
    """Terminal colors for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name: str):
    """Print test header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_response(data: Dict, indent: int = 2):
    """Print JSON response"""
    print(f"{Colors.YELLOW}{json.dumps(data, indent=indent, default=str)}{Colors.END}")

def test_health_check() -> bool:
    """Test health check endpoint"""
    print_test("Health Check - GET /")
    
    try:
        resp = requests.get(f"{BASE_URL}/")
        
        if resp.status_code == 200:
            print_success(f"Status: {resp.status_code}")
            print_response(resp.json())
            return True
        else:
            print_error(f"Unexpected status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_create_violation() -> Optional[str]:
    """Test creating a violation"""
    print_test("Create Violation - POST /violations")
    
    violation_id = f"V-TEST-{int(datetime.now().timestamp())}"
    
    violation = {
        "violation_id": violation_id,
        "type": "helmetless",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "camera_id": "CAM_01",
        "location": "Test Junction A",
        "plate": "KA01TEST123",
        "confidence": 0.95,
        "media": {
            "context_img": f"output/violations/{violation_id}/context.jpg",
            "plate_img": f"output/violations/{violation_id}/plate.jpg",
            "clip_video": f"output/violations/{violation_id}/clip.mp4"
        },
        "extra": {"test": True, "vehicle_type": "motorcycle"}
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/violations", json=violation)
        
        if resp.status_code == 200:
            print_success(f"Created violation: {violation_id}")
            print_response(resp.json())
            return violation_id
        else:
            print_error(f"Failed with status: {resp.status_code}")
            print(resp.text)
            return None
    except Exception as e:
        print_error(f"Request failed: {e}")
        return None

def test_list_violations() -> bool:
    """Test listing all violations"""
    print_test("List All Violations - GET /violations")
    
    try:
        resp = requests.get(f"{BASE_URL}/violations")
        
        if resp.status_code == 200:
            violations = resp.json()
            print_success(f"Found {len(violations)} violations")
            
            if violations:
                print("\nFirst violation:")
                print_response(violations[0])
            
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_filter_by_type() -> bool:
    """Test filtering violations by type"""
    print_test("Filter by Type - GET /violations?vtype=helmetless")
    
    try:
        resp = requests.get(f"{BASE_URL}/violations?vtype=helmetless")
        
        if resp.status_code == 200:
            violations = resp.json()
            print_success(f"Found {len(violations)} helmetless violations")
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_filter_by_confidence() -> bool:
    """Test filtering by confidence threshold"""
    print_test("Filter by Confidence - GET /violations?min_confidence=0.9")
    
    try:
        resp = requests.get(f"{BASE_URL}/violations?min_confidence=0.9")
        
        if resp.status_code == 200:
            violations = resp.json()
            print_success(f"Found {len(violations)} high-confidence violations (≥0.9)")
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_get_single_violation(violation_id: str) -> bool:
    """Test getting a single violation by ID"""
    print_test(f"Get Single Violation - GET /violations/{violation_id}")
    
    try:
        resp = requests.get(f"{BASE_URL}/violations/{violation_id}")
        
        if resp.status_code == 200:
            print_success(f"Retrieved violation: {violation_id}")
            print_response(resp.json())
            return True
        elif resp.status_code == 404:
            print_error(f"Violation not found: {violation_id}")
            return False
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_review_violation(violation_id: str) -> bool:
    """Test reviewing a violation"""
    print_test(f"Review Violation - PATCH /violations/{violation_id}/review")
    
    try:
        resp = requests.patch(
            f"{BASE_URL}/violations/{violation_id}/review",
            params={
                "approved": True,
                "notes": "Verified - Clear helmet violation detected"
            }
        )
        
        if resp.status_code == 200:
            print_success(f"Reviewed violation: {violation_id}")
            print_response(resp.json())
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_filter_reviewed() -> bool:
    """Test filtering reviewed violations"""
    print_test("Filter Reviewed - GET /violations?reviewed=true")
    
    try:
        resp = requests.get(f"{BASE_URL}/violations?reviewed=true")
        
        if resp.status_code == 200:
            violations = resp.json()
            print_success(f"Found {len(violations)} reviewed violations")
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_statistics() -> bool:
    """Test statistics endpoint"""
    print_test("Statistics - GET /stats")
    
    try:
        resp = requests.get(f"{BASE_URL}/stats")
        
        if resp.status_code == 200:
            print_success("Retrieved statistics")
            print_response(resp.json())
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_import_from_disk() -> bool:
    """Test import from disk endpoint"""
    print_test("Import from Disk - POST /violations/import_from_disk")
    
    try:
        resp = requests.post(f"{BASE_URL}/violations/import_from_disk")
        
        if resp.status_code == 200:
            result = resp.json()
            print_success(f"Imported {result.get('imported', 0)} violations")
            print_response(result)
            return True
        else:
            print_error(f"Failed with status: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    print(f"\n{Colors.GREEN}{'='*60}")
    print("🧪 AI Traffic Violation Backend - Test Suite")
    print(f"{'='*60}{Colors.END}")
    print(f"Backend URL: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Import existing violations
    results.append(("Import from Disk", test_import_from_disk()))
    
    # Test 3: Create new violation
    violation_id = test_create_violation()
    results.append(("Create Violation", violation_id is not None))
    
    # Test 4: List all violations
    results.append(("List Violations", test_list_violations()))
    
    # Test 5: Filter by type
    results.append(("Filter by Type", test_filter_by_type()))
    
    # Test 6: Filter by confidence
    results.append(("Filter by Confidence", test_filter_by_confidence()))
    
    # Test 7: Get single violation
    if violation_id:
        results.append(("Get Single Violation", test_get_single_violation(violation_id)))
        
        # Test 8: Review violation
        results.append(("Review Violation", test_review_violation(violation_id)))
        
        # Test 9: Filter reviewed
        results.append(("Filter Reviewed", test_filter_reviewed()))
    
    # Test 10: Statistics
    results.append(("Statistics", test_statistics()))
    
    # Print summary
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Test Summary")
    print(f"{'='*60}{Colors.END}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    success_rate = (passed / total * 100) if total > 0 else 0
    
    if passed == total:
        print(f"{Colors.GREEN}🎉 All tests passed! ({passed}/{total}) - {success_rate:.0f}%{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  Some tests failed ({passed}/{total}) - {success_rate:.0f}%{Colors.END}")
    
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Tests interrupted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Test suite failed: {e}{Colors.END}")
