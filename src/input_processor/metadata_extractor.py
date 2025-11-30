import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import imagehash
import piexif

class MetadataExtractor:
    """
    Extracts metadata (EXIF, GPS, Timestamp) and computes perceptual hash for images.
    """
    
    @staticmethod
    def get_exif_data(image: Image.Image) -> Dict[str, Any]:
        """
        Returns a dictionary from the exif data of an PIL Image item. 
        Also converts the GPS Tags
        """
        exif_data = {}
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        return exif_data

    @staticmethod
    def _get_if_exist(data: Dict, key: str) -> Any:
        return data.get(key)

    @staticmethod
    def _convert_to_degrees(value: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> float:
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
        """
        d = value[0]
        m = value[1]
        s = value[2]
        return d + (m / 60.0) + (s / 3600.0)

    @staticmethod
    def get_lat_lon(exif_data: Dict) -> Optional[Tuple[float, float]]:
        """
        Returns the latitude and longitude, if available, from the provided exif_data
        """
        lat = None
        lon = None

        if "GPSInfo" in exif_data:
            gps_info = exif_data["GPSInfo"]

            gps_latitude = MetadataExtractor._get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = MetadataExtractor._get_if_exist(gps_info, "GPSLatitudeRef")
            gps_longitude = MetadataExtractor._get_if_exist(gps_info, "GPSLongitude")
            gps_longitude_ref = MetadataExtractor._get_if_exist(gps_info, "GPSLongitudeRef")

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = MetadataExtractor._convert_to_degrees(gps_latitude)
                if gps_latitude_ref != "N":
                    lat = 0 - lat

                lon = MetadataExtractor._convert_to_degrees(gps_longitude)
                if gps_longitude_ref != "E":
                    lon = 0 - lon

        if lat is not None and lon is not None:
            return lat, lon
        return None

    @staticmethod
    def get_timestamp(exif_data: Dict) -> Optional[datetime]:
        """
        Extracts the timestamp from EXIF data.
        """
        # Try different date tags
        date_str = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime") or exif_data.get("DateTimeDigitized")
        
        if date_str:
            try:
                # Standard EXIF date format: YYYY:MM:DD HH:MM:SS
                return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        return None

    @staticmethod
    def get_orientation(image: Image.Image) -> str:
        """
        Determines if image is landscape or portrait based on dimensions.
        """
        width, height = image.size
        if width >= height:
            return "landscape"
        return "portrait"

    @staticmethod
    def compute_hash(image: Image.Image) -> str:
        """
        Computes perceptual hash for duplicate detection.
        """
        return str(imagehash.phash(image))

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        """
        Main entry point to extract all metadata from an image file.
        """
        try:
            with Image.open(image_path) as img:
                # Handle EXIF orientation if present (to ensure correct width/height)
                # This is a simplified handling; for full robustness we might need ImageOps.exif_transpose
                
                exif_data = cls.get_exif_data(img)
                timestamp = cls.get_timestamp(exif_data)
                gps = cls.get_lat_lon(exif_data)
                orientation = cls.get_orientation(img)
                img_hash = cls.compute_hash(img)
                
                # Fallback for timestamp if not in EXIF: use file modification time
                if not timestamp:
                    timestamp = datetime.fromtimestamp(os.path.getmtime(image_path))

                return {
                    "filepath": str(image_path),
                    "filename": os.path.basename(image_path),
                    "timestamp": timestamp,
                    "gps_coords": gps,
                    "orientation": orientation,
                    "hash": img_hash,
                    "width": img.width,
                    "height": img.height,
                    "camera_model": exif_data.get("Model", "Unknown")
                }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                "filepath": str(image_path),
                "filename": os.path.basename(image_path),
                "error": str(e)
            }
