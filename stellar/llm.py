import os
import json
import logging
from pathlib import Path, PurePosixPath
import shutil
import tempfile

logger = logging.getLogger(__name__)


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

def call_cline(prompt: str, mount_path: str):
    import docker
    
    client = docker.from_env()
    
    assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY is not set"
    assert ANTHROPIC_API_KEY is not None, "ANTHROPIC_API_KEY is not set"
    
    out = client.containers.run(
        "stchain:2.0", 
        f'python agent_core.py "{prompt}"', 
        remove=True,
        volumes=[
            f"{mount_path}:/output",
            f'{mount_path}/log:/root/.cline/data/tasks',
        ],
        working_dir="/workspace",
        environment=[
            "AGENT_EXELIMIT=1800",
            f"GOOGLE_API_KEY={GOOGLE_API_KEY}",
            f"ANTHROPIC_API_KEY={ANTHROPIC_API_KEY}"
        ],
        device_requests=[
            docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]
    )
    if len(out.decode('utf-8')) != 0:
        return False, "============= code gen error ============="
    else:
        return True, out.decode('utf-8')

def cline_main(prompt: str):
    prompt = 'I want to create a CNN model YOLOX and COCO, reference with https://github.com/bubbliiiing/yolox-pytorch'
    
    with tempfile.TemporaryDirectory() as mount_path:
        call_cline(prompt, mount_path)

def locate_model_codebase(output_dir: str):
    output_dir = Path(output_dir)
    files = list(output_dir.glob("*.zip"))
    assert len(files) > 0, "No zip files found"
    assert len(files) == 1, "Only one zip file is allowed"
    return files[0]

def extract_model_codebase(zip_file: Path):
    logger.info(f"Processing zip file: {zip_file}")
    
    model_fn_str = ""
    dataloader_fn_str = ""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.unpack_archive(zip_file, tmp_dir, 'zip')
        
        def get_module_name(file: Path):
            return PurePosixPath(file.relative_to(tmp_dir)).stem.replace("/", ".")
        
        for file in Path(tmp_dir).glob("**/metadata.json"):
            with open(file, "r") as f:
                metadata = json.load(f)
                if "model_fn" in metadata:
                    model_fn_str = metadata["model_fn"]
                if "dataloader_fn" in metadata:
                    dataloader_fn_str = metadata["dataloader_fn"]
                if model_fn_str != "" and dataloader_fn_str != "":
                    logger.info(f"Found model function and dataloader function in metadata.json: {model_fn_str}, {dataloader_fn_str}")
                    return model_fn_str, dataloader_fn_str
        
        for file in Path(tmp_dir).glob("**/*.py"):
            with open(file, "r") as f:
                code = f.read()
                if "def model_fn(" in code:
                    model_fn_str = get_module_name(file) + ".model_fn"
                if "def dataloader_fn(" in code:
                    dataloader_fn_str = get_module_name(file) + ".dataloader_fn"
                if model_fn_str != "" and dataloader_fn_str != "":
                    logger.info(f"Found model function and dataloader function in .py file: {model_fn_str}, {dataloader_fn_str}")
                    return model_fn_str, dataloader_fn_str
    
    assert model_fn_str != "", "Model function not found"
    assert dataloader_fn_str != "", "Dataloader function not found"
        
    return model_fn_str, dataloader_fn_str

def save_model_codebase(model_name: str, source_file: str, target_dir: str):
    model_fn_str, dataloader_fn_str = extract_model_codebase(source_file)
    logger.info(f"Model function: {model_fn_str}")
    logger.info(f"Dataloader function: {dataloader_fn_str}")
    
    save_dir = Path(target_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "model_fn": model_fn_str,
            "dataloader_fn": dataloader_fn_str
        }, f)
    shutil.copy(source_file, save_dir / "model.zip")
    return str(save_dir.absolute())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mount_path = "/mnt/d/ST/03_FL/1114_shaoyu/STChain_v1/holder"
    zip_file = locate_model_codebase(mount_path)
    logger.info(f"Located zip file: {zip_file}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = save_model_codebase("test", zip_file, tmp_dir)
        logger.info(f"Saved directory: {save_dir}")
        logger.info(f"Saved files: {list(Path(save_dir).glob('**/*'))}")
        with open(Path(save_dir) / "metadata.json", "r") as f:
            metadata = json.load(f)
            logger.info(f"Metadata: {metadata}")
