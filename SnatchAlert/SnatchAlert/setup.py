"""
Quick setup script for SnatchAlert
Run with: python setup.py
"""
import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              SnatchAlert Setup Script                     ║
    ║         Crime Reporting & Tracking System                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  Prerequisites Check:")
    print("   - Python 3.10+ installed")
    print("   - PostgreSQL running")
    print("   - Database 'snatchalertdb' created")
    print("   - Virtual environment activated (recommended)")
    
    response = input("\n✓ All prerequisites met? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ Please complete prerequisites first. See QUICKSTART.md")
        sys.exit(1)
    
    print("\n🚀 Starting setup process...")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n⚠️  Dependency installation failed. Try manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Make migrations
    if not run_command("python manage.py makemigrations", "Creating migrations"):
        print("\n⚠️  Migration creation failed.")
        sys.exit(1)
    
    # Run migrations
    if not run_command("python manage.py migrate", "Running migrations"):
        print("\n⚠️  Migration failed. Check database connection.")
        sys.exit(1)
    
    # Load seed data
    response = input("\n📊 Load sample data? (y/n): ")
    if response.lower() == 'y':
        if run_command("python manage.py seed_data", "Loading seed data"):
            print("\n✅ Sample data loaded successfully!")
            print("\n📝 Test Credentials:")
            print("   Admin: username=admin, password=admin123")
            print("   Authority: username=police_officer, password=police123")
            print("   User: username=john_doe, password=user123")
    
    # Create superuser
    response = input("\n👤 Create custom superuser? (y/n): ")
    if response.lower() == 'y':
        run_command("python manage.py createsuperuser", "Creating superuser")
    
    print("""
    
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              ✅ Setup Complete!                           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🚀 Next Steps:
    
    1. Start the development server:
       python manage.py runserver
    
    2. Access the application:
       - API Docs: http://localhost:8000/api/docs/
       - Admin Panel: http://localhost:8000/admin/
       - API Base: http://localhost:8000/api/
    
    3. Test the API:
       - Import SnatchAlert_API_Collection.json into Postman
       - Or use the Swagger UI at /api/docs/
    
    📚 Documentation:
       - README.md - Complete documentation
       - QUICKSTART.md - Quick start guide
       - API docs at /api/docs/
    
    💡 Need help? Check the documentation or visit the admin panel.
    
    Happy coding! 🎉
    """)

if __name__ == "__main__":
    main()
