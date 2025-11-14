# FILE: utils/file_utils.py
"""
存放文件操作相关的辅助函数。
"""
from pathlib import Path

def create_subset_file(source_path: Path, dest_path: Path, limit_mb: int):
    """
    以内存高效的方式从源文件创建一个较小的子集文件。
    Args:
        source_path: 原始数据文件的路径。
        dest_path: 要创建的子集文件的路径。
        limit_mb: 子集文件的大小上限（MB）。
    """
    limit_bytes = limit_mb * 1024 * 1024
    buffer_size = 4 * 1024 * 1024  # 4MB 缓冲区
    bytes_written = 0

    with open(source_path, 'r', encoding='utf-8') as f_in, \
         open(dest_path, 'w', encoding='utf-8') as f_out:
        while bytes_written < limit_bytes:
            chunk = f_in.read(buffer_size)
            if not chunk:
                break  # 文件已读完

            chunk_bytes = chunk.encode('utf-8')
            # 如果加上这个块会超过限制，就只写入需要的部分
            if bytes_written + len(chunk_bytes) > limit_bytes:
                # 估算需要截断的字符数
                # 这不是完全精确的，但对于分词器训练来说足够好
                bytes_to_keep = limit_bytes - bytes_written
                # 避免除以零
                if len(chunk_bytes) > 0:
                    chars_to_keep = int(bytes_to_keep * (len(chunk) / len(chunk_bytes)))
                    chunk = chunk[:chars_to_keep]
                else:
                    break
                f_out.write(chunk)
                break

            f_out.write(chunk)
            bytes_written += len(chunk_bytes)

# END OF FILE: utils/file_utils.py