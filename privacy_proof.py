import requests
import json

class PrivacyProofAPI:
    def __init__(self, base_url="http://17164778.r9.cpolar.top"):
        self.base_url = base_url.rstrip("/")

    def gen_pedersen_proof(self, name, K, Q, data):
        """{
            "name": //知识库名，例如“law”
            "K": ,
            "Q": , //query向量
            "data": //数据块
            }
        """
        url = f"{self.base_url}/gen_pedersen_merkle_proof"
        payload = {"name": name, "K": K, "Q": Q, "data": data}
        return self._post(url, payload)

    def gen_groth_proof(self, name, K, Q, data):
        """{
            "name": //知识库名，例如“law”
            "K": ,
            "Q": ,
            "data": //数据块
            }
        """
        url = f"{self.base_url}/gen_groth_merkle_proof"
        payload = {"name": name, "K": K, "Q": Q, "data": data}
        return self._post(url, payload)

    def verify_pedersen_proof(self, proof_id):
        """{
            "ProofId": 生成证明时会返回proofid，直接copy过来
            }
        """
        url = f"{self.base_url}/verify_pedersen_merkle_proof"
        payload = {"ProofId": proof_id}
        return self._post(url, payload)

    def verify_groth_proof(self, proof_id):
        """{
            "ProofId": 生成证明时会返回proofid，直接copy过来
            }
        """
        url = f"{self.base_url}/verify_groth_merkle_proof"
        payload = {"ProofId": proof_id}
        return self._post(url, payload)

    def gen_pog(self, Q, K, data):
        """{
            "Q":
            "K":
            "data": //同上
            }
        """
        url = f"{self.base_url}/gen_PoG"
        payload = {"Q": Q, "K": K, "data": data}
        return self._post(url, payload)

    def verify_pog(self, por_id, pog_id):
        """{
            "por_id": 1001,  // PoR的ID
            "pog_id": 2001   // PoG的ID
        }
        """
        url = f"{self.base_url}/verify_PoG"
        payload = {"por_id": por_id, "pog_id": pog_id}
        return self._post(url, payload)

    def _post(self, url, payload):
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[Error] POST {url} failed: {e}")
            return None
