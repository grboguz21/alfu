"""
MinIO Manager Module
Uploads images to MinIO and manages paths

Usage:
    from minio_manager import MinIOManager

    minio = MinIOManager(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        branch_name="Tepe Prime"
    )

    # Upload image
    path = minio.upload_image(
        image=frame,
        folder="Table Occupancy Analysis",
        filename="Report1.jpg"
    )
    # Returns: "Tepe Prime/26-11-2025/Table Occupancy Analysis/Report1.jpg"
"""

import cv2
import io
import os
import urllib3
from datetime import datetime
from minio import Minio
from minio.error import S3Error
from typing import Optional



class MinIOManager:
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 branch_name: str = None, secure: bool = False, 
                 bucket_name: str = "ai-outputs"):
        """
        Initialize MinIO Manager

        Args:
            endpoint: MinIO endpoint (e.g. "localhost:9000")
            access_key: Access key
            secret_key: Secret key
            branch_name: Branch name (e.g. "Tepe Prime")
            secure: Use HTTPS
            bucket_name: Default bucket
        """
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.branch_name = branch_name
        
        try:
            # ✅ TIMEOUT ADDED: 5s connect, 10s read
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=5.0, read=10.0),
                retries=urllib3.Retry(total=3, backoff_factor=0.5)
            )
            
            self.client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                http_client=http_client  # ✅ With timeout
            )
            print(f"✅ MinIO connection established: {endpoint} (timeout: 5s/10s)")
        except Exception as e:
            print(f"❌ MinIO connection error: {e}")
            self.client = None
    
    
    def upload_image(self, image, folder: str, filename: str,
                     bucket_name: str = None) -> Optional[str]:
        """
        Upload image to MinIO

        Args:
            image: cv2 image (BGR numpy array)
            folder: Module folder (e.g. "Table Occupancy Analysis")
            filename: File name (e.g. "Report1.jpg")
            bucket_name: Bucket (None uses default)

        Returns:
            full_object_path: "Branch Name/DD-MM-YYYY/folder/filename" or None
        """
        if self.client is None:
            print("❌ No MinIO client")
            return None

        if image is None or image.size == 0:
            print("❌ Empty image, cannot upload")
            return None
        
        bucket = bucket_name or self.bucket_name
        
        try:
            # Encode image
            is_success, buffer = cv2.imencode(".jpg", image)
            if not is_success:
                print("❌ Image encoding failed")
                return None

            # Convert to BytesIO
            byte_io = io.BytesIO(buffer)

            # Build path: "Branch Name/DD-MM-YYYY/folder/filename"
            today_str = datetime.now().strftime("%d-%m-%Y")

            if self.branch_name:
                full_object_path = f"{self.branch_name}/{today_str}/{folder}/{filename}"
            else:
                full_object_path = f"{today_str}/{folder}/{filename}"

            # Upload
            self.client.put_object(
                bucket_name=bucket,
                object_name=full_object_path,
                data=byte_io,
                length=len(byte_io.getvalue()),
                content_type="image/jpeg"
            )
            
            print(f"✅ Uploaded: {full_object_path}")
            return full_object_path

        except urllib3.exceptions.TimeoutError:
            print(f"❌ MinIO timeout error (connection timed out)")
            return None
        except S3Error as e:
            print(f"❌ MinIO S3 error: {e}")
            return None
        except Exception as e:
            print(f"❌ MinIO upload error: {e}")
            return None
    
    
    def upload_report(self, image, folder: str,
                      bucket_name: str = None) -> Optional[str]:
        """
        Uploads periodic report image.
        Always overwrites the same file (snapshot.jpg).
        """
        return self.upload_image(image, folder, "snapshot.jpg", bucket_name)

    def upload_alert(self, image, folder: str,
                     bucket_name: str = None,
                     prefix: str = "") -> Optional[str]:
        """
        Uploads alert image with a unique timestamp filename.
        E.g. alert_14-32-05.jpg
        """
        filename = prefix + datetime.now().strftime("%H-%M-%S.jpg")
        return self.upload_image(image, folder, filename, bucket_name)

    def get_last_alert_index(self, folder: str, prefix: str = "Alert") -> int:
        """
        Find the last alert index in the folder

        Args:
            folder: Module folder (e.g. "Table Occupancy Analysis")
            prefix: File prefix (e.g. "Alert")

        Returns:
            Last index (e.g. returns 5 if Alert5.jpg exists)
        """
        if self.client is None:
            return 0
        
        today_str = datetime.now().strftime("%d-%m-%Y")

        # Build path: "Branch Name/DD-MM-YYYY/folder/prefix"
        if self.branch_name:
            search_prefix = f"{self.branch_name}/{today_str}/{folder}/{prefix}"
        else:
            search_prefix = f"{today_str}/{folder}/{prefix}"
        
        max_index = 0
        try:
            objects = self.client.list_objects(
                self.bucket_name, 
                prefix=search_prefix, 
                recursive=True
            )
            
            for obj in objects:
                name = os.path.basename(obj.object_name)
                if name.startswith(prefix) and name.endswith(".jpg"):
                    # "Alert5.jpg" -> "5"
                    num_part = name.replace(prefix, "").replace(".jpg", "")
                    if num_part.isdigit():
                        max_index = max(max_index, int(num_part))
        
        except urllib3.exceptions.TimeoutError:
            print(f"❌ MinIO timeout error (list_objects)")
        except Exception as e:
            print(f"❌ MinIO count error: {e}")
        
        return max_index
    
    
    def generate_alert_filename(self, folder: str, prefix: str = "Alert") -> tuple[str, int]:
        """
        Auto-generate alert filename

        Args:
            folder: Folder name
            prefix: File prefix

        Returns:
            (filename, index): ("Alert6.jpg", 6)
        """
        last_index = self.get_last_alert_index(folder, prefix)
        next_index = last_index + 1
        filename = f"{prefix}{next_index}.jpg"
        return filename, next_index


