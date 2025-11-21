from .http_utils import send_request, send_files
import json
import re
import base64
from PIL import Image
import io
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading 

def transformation_matrix(trans, quat):
    # Convert quaternion to 3x3 rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = trans
    return T

class Domain:
    def __init__(self, domain_config, portal_conversion_matrix=None):
        """
        Initialize the Domain object.
        
        :param domain_config: dict type that contains domain configurations.
        """
        self.domain_id = domain_config["domain_id"]
        self.account = domain_config["posemesh_account"]
        self.password = domain_config["posemesh_password"]
        self.map_endpoint = domain_config["map_endpoint"]
        
        self._posemesh_token = ''
        self._dds_token = ''
        self._domain_info = {}
        self._domain_server = None

        self._portals = {}
        self._portal_conversion_matrix = portal_conversion_matrix

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_every_hour, daemon=True)
        self._thread.start()

    def _run_every_hour(self):
        while not self._stop_event.is_set():
            self.auth()
            self.fetch_portal_poses()
            # Wait one hour (3600 seconds) or until the stop event is set
            self._stop_event.wait(timeout=3600)

    def auth(self):
        # Auth User Posemesh
        url1 = "https://api.posemesh.org/user/login"
        headers1 = {'Content-Type': 'application/json',
                    'Accept': 'application/json'}
        body1 = {'email': self.account,
                 'password': self.password}
        
        ret1, response1 = send_request('POST', url1, headers1, body1)
        if not ret1:
            return False, 'Failed to authenticate posemesh account'

        rep_json1 = json.loads(response1.text)
        self._posemesh_token = rep_json1['access_token']

        # Auth DDS
        url2 = "https://api.posemesh.org/service/domains-access-token"
        headers2 = {'Accept': 'application/json',
                    'Authorization': f"Bearer {self._posemesh_token}"}
        ret2, response2 = send_request('POST', url2, headers2)
        if not ret2:
            return False, 'Failed to authenticate domain dds'
        rep_json2 = json.loads(response2.text)
        self._dds_token = rep_json2['access_token']

        # Auth Domain
        url3 = f"https://dds.posemesh.org/api/v1/domains/{self.domain_id}/auth"
        headers3 = {'Accept': 'application/json',
                    'Authorization': f"Bearer {self._dds_token}"}
        ret3, response3 = send_request('POST', url3, headers3)
        if not ret3:
            return False, 'Failed to authenticate domain access'
        self._domain_info = json.loads(response3.text)
        self._domain_server = self._domain_info["domain_server"]["url"]

        return True, ''

    def fetch_portal_poses(self):
        url = f"{self._domain_info['domain_server']['url']}/api/v1/domains/{self._domain_info['id']}/lighthouses"
        
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}
        ret, response = send_files('GET', url, headers)
        if not ret:
            return False, 'Failed to fetch the portals information'
        
        response_json = json.loads(response.text)
        
        for qr in response_json['poses']:

            trans = np.array([qr['px'], qr['py'], qr['pz']])
            quat = np.array([qr['rx'], qr['ry'], qr['rz'], qr['rw']])

            T_domain_portal = transformation_matrix(trans, quat)
            if self._portal_conversion_matrix is not None:
                T_domain_portal = self._portal_conversion_matrix @ T_domain_portal

            portal = {
                "short_id": qr["short_id"],
                "size": float(qr['reported_size']) / 100.0,
                "pose": T_domain_portal
            }
            self._portals[qr["short_id"]] = portal
        
        return True, ''
    
    def portals(self):
        return self._portals

    def get_map(self, resolution=20):
        method = 'POST'

        url = self.map_endpoint
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}

        body = {
            'domainId': self.domain_id,
            'domainServerUrl': self._domain_server,
            'height': 0.1,
            'pixelsPerMeter': resolution,
            'fileType': 'png',
        }

        success, response = send_request(method, url, headers, body)
        if not success:
            return None, None
        raw_data = response.text

        # Split the data using the boundary marker
        boundary = raw_data.split("\n", 1)[0].strip()
        parts = raw_data.split(boundary)

        # Initialize placeholders for the image and YAML data
        image_data = None
        yaml_data = None

        # Iterate through each part of the form-data
        for part in parts:
            if "name=\"img\"" in part:
                # Extract and decode the base64 image data, handle newlines
                image_data_match = re.search(r"name=\"img\"\s*\n([a-zA-Z0-9+/=\n]+)", part)
                if image_data_match:
                    # Remove any newlines or spaces in the base64-encoded data
                    encoded_image = "".join(image_data_match.group(1).splitlines())
                    image_data = base64.b64decode(encoded_image)
            elif "name=\"yaml\"" in part:
                # Extract the YAML content
                yaml_data_match = re.search(r"name=\"yaml\"\s*\n(.+)", part, re.DOTALL)
                if yaml_data_match:
                    yaml_data = yaml_data_match.group(1).strip()

        image = Image.open(io.BytesIO(image_data))
        yaml_dict = yaml.safe_load(yaml_data)
        return image, yaml_dict