from pathlib import Path

from PIL import Image


def main():
    project_root = Path(__file__).resolve().parents[1]
    installer_dir = project_root / "installer"
    logo_png_path = installer_dir / "logo.png"
    logo_ico_path = installer_dir / "logo.ico"
    wizard_bmp_path = installer_dir / "wizard.bmp"
    wizard_small_bmp_path = installer_dir / "wizard_small.bmp"

    if not logo_png_path.exists():
        print(f"Error: {logo_png_path} does not exist.")
        return

    # Load original generated image
    img = Image.open(logo_png_path).convert("RGBA")
    if img.width != img.height:
        side = max(img.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
        img = square

    # 1. Save as Multi-Resolution Windows ICO
    print("Generating logo.ico...")
    icon_sizes = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(logo_ico_path, format="ICO", sizes=icon_sizes)
    print("logo.ico created.")

    # 2. Save as Wizard Small BMP (Header image: 55x55)
    print("Generating wizard_small.bmp...")
    small_img = img.resize((55, 55), Image.Resampling.LANCZOS)
    # Convert to RGB because BMP doesn't support transparency/alpha channel in standard Inno Setup
    small_rgb = Image.new("RGB", (55, 55), (26, 26, 32)) # background #1a1a20
    if img.mode == 'RGBA':
        small_rgb.paste(small_img, (0, 0), small_img)
    else:
        small_rgb.paste(small_img, (0, 0))
    small_rgb.save(wizard_small_bmp_path, format="BMP")
    print("wizard_small.bmp created.")

    # 3. Save as Wizard Sidebar BMP (Sidebar image: 164x314)
    print("Generating wizard.bmp...")
    # Create a nice dark background strip
    sidebar = Image.new("RGB", (164, 314), (26, 26, 32)) # background #1a1a20
    
    # Resize the logo to fit nicely in the sidebar (e.g. 110x110)
    logo_resized = img.resize((110, 110), Image.Resampling.LANCZOS)
    
    # Paste logo in the upper center of the sidebar
    paste_x = (164 - 110) // 2
    paste_y = 60 # 60 pixels from the top
    if img.mode == 'RGBA':
        sidebar.paste(logo_resized, (paste_x, paste_y), logo_resized)
    else:
        sidebar.paste(logo_resized, (paste_x, paste_y))
        
    sidebar.save(wizard_bmp_path, format="BMP")
    print("wizard.bmp created.")
    print("All Inno Setup graphic assets successfully generated!")

if __name__ == "__main__":
    main()
