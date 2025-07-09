# AWK Scaling Recommendations

This document outlines various approaches to improve the performance and scalability of AWK subprocess calls in Python, particularly for large-scale text processing tasks like fuel receipt extraction.

## Current Implementation

The basic AWK subprocess approach:

```python
import subprocess

def run_awk(script, input_text):
    result = subprocess.run(
        ['awk', script],
        input=input_text,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

# Example usage
text = "John 25 Engineer\nJane 30 Doctor"
result = run_awk('{print $1, $3}', text)
```

## Scaling Improvements

### 1. Connection Pooling & Persistent Processes

Reduce process creation overhead by maintaining a pool of persistent AWK processes:

```python
import subprocess
from threading import Lock
from queue import Queue

class AWKProcessor:
    def __init__(self, pool_size=4):
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()
        
        # Pre-spawn AWK processes
        for _ in range(pool_size):
            proc = subprocess.Popen(
                ['awk', '-f', '/dev/stdin'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.pool.put(proc)
    
    def run_awk(self, script, input_text):
        proc = self.pool.get()
        try:
            stdout, stderr = proc.communicate(f"{script}\n{input_text}")
            return stdout.strip()
        finally:
            self.pool.put(proc)
```

**Benefits:**
- Eliminates process creation overhead
- Maintains warm AWK interpreters
- Thread-safe process sharing

### 2. Batch Processing

Process multiple AWK operations simultaneously:

```python
def run_awk_batch(scripts_and_inputs):
    """Process multiple AWK operations in one call"""
    processes = []
    
    for script, input_text in scripts_and_inputs:
        proc = subprocess.Popen(
            ['awk', script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        proc.stdin.write(input_text)
        proc.stdin.close()
        processes.append(proc)
    
    return [proc.stdout.read().strip() for proc in processes]
```

**Benefits:**
- Parallel execution of multiple AWK scripts
- Better resource utilization
- Reduced total processing time

### 3. Async Processing

Use asyncio for non-blocking AWK operations:

```python
import asyncio

async def run_awk_async(script, input_text):
    proc = await asyncio.create_subprocess_exec(
        'awk', script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = await proc.communicate(input_text)
    return stdout.strip()

# Usage
async def process_many():
    tasks = [run_awk_async(script, text) for script, text in data]
    return await asyncio.gather(*tasks)
```

**Benefits:**
- Non-blocking I/O operations
- Efficient handling of many concurrent requests
- Better CPU utilization during I/O waits

### 4. File-Based Processing (Recommended for Large Data)

Most efficient approach for large datasets:

```python
def run_awk_file(script, input_file, output_file):
    """Most efficient for large datasets"""
    subprocess.run([
        'awk', script, input_file
    ], stdout=open(output_file, 'w'))

def run_awk_pipe(script, input_files):
    """Stream processing multiple files"""
    cmd = ['awk', script] + input_files
    return subprocess.run(cmd, capture_output=True, text=True).stdout
```

**Benefits:**
- Leverages AWK's native file processing capabilities
- Minimal memory overhead
- Optimal for large document processing

### 5. Memory-Mapped Files

For very large files that don't fit in memory:

```python
import mmap

def run_awk_mmap(script, large_file):
    with open(large_file, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0) as mm:
            proc = subprocess.run(
                ['awk', script],
                input=mm,
                capture_output=True,
                text=True
            )
            return proc.stdout
```

**Benefits:**
- Efficient handling of very large files
- Reduced memory footprint
- OS-level memory management

### 6. Compiled AWK Scripts

Pre-compile and cache AWK scripts:

```python
class CompiledAWKScript:
    def __init__(self, script):
        self.script_file = f"/tmp/awk_script_{id(self)}.awk"
        with open(self.script_file, 'w') as f:
            f.write(script)
    
    def run(self, input_text):
        return subprocess.run(
            ['awk', '-f', self.script_file],
            input=input_text,
            text=True,
            capture_output=True
        ).stdout.strip()
    
    def __del__(self):
        try:
            os.unlink(self.script_file)
        except:
            pass
```

**Benefits:**
- Avoid script parsing overhead
- Reusable compiled scripts
- Better performance for repeated operations

## Recommendations by Use Case

### For Fuel Receipt Processing

**Recommended approach:** Combination of **File-Based Processing** + **Connection Pooling**

```python
class FuelReceiptProcessor:
    def __init__(self):
        self.awk_pool = AWKProcessor(pool_size=4)
        self.compiled_scripts = {}
    
    def process_receipt_batch(self, receipt_files):
        # Use file-based processing for large batches
        return run_awk_pipe(self.fuel_extraction_script, receipt_files)
    
    def process_single_receipt(self, receipt_text):
        # Use pooled processes for individual receipts
        return self.awk_pool.run_awk(self.fuel_extraction_script, receipt_text)
```

### For Real-time Processing

**Recommended approach:** **Async Processing** + **Connection Pooling**

### For Batch Processing

**Recommended approach:** **File-Based Processing** + **Batch Processing**

### For Memory-Constrained Environments

**Recommended approach:** **Memory-Mapped Files** + **Compiled Scripts**

## Performance Considerations

1. **Process Creation Overhead**: Most expensive operation - use pooling
2. **Memory Usage**: File-based processing uses less memory than string-based
3. **I/O Bottlenecks**: Async processing helps with I/O-bound operations
4. **CPU Utilization**: Batch processing maximizes CPU usage
5. **Script Parsing**: Compiled scripts avoid repeated parsing overhead

## Integration with Existing Codebase

To integrate these improvements with the current `awk_extractor.py`:

1. Replace the current simulation with actual AWK calls
2. Use file-based processing for large document batches
3. Implement connection pooling for interactive processing
4. Cache compiled AWK scripts for frequently used patterns

This approach would provide significant performance improvements while maintaining the familiar AWK syntax and processing model.

## Kubeflow Pipelines (KFP) Considerations

When running in a Kubeflow Pipelines environment, the recommendations change significantly due to the containerized, distributed nature of the platform:

### KFP-Specific Constraints

1. **Container Lifecycle**: Each pipeline component runs in its own container with limited lifecycle
2. **Resource Limits**: CPU and memory are constrained by Kubernetes resource quotas
3. **Ephemeral Storage**: Local storage is temporary and lost between component runs
4. **Network Isolation**: Limited network access between components
5. **No Persistent Processes**: Cannot maintain long-running processes between invocations

### Recommended Approaches for KFP

#### 1. **Stateless File-Based Processing** (Primary Recommendation)

```python
def kfp_awk_processor(input_path: str, output_path: str, script: str):
    """KFP component for AWK processing - stateless and containerized"""
    import subprocess
    from pathlib import Path
    
    # Use file-based processing exclusively
    result = subprocess.run([
        'awk', script, input_path
    ], capture_output=True, text=True, check=True)
    
    # Write results to output path for next component
    Path(output_path).write_text(result.stdout)
    
    return output_path
```

#### 2. **Batch Processing with Resource Optimization**

```python
def kfp_batch_awk_processor(input_files: List[str], output_dir: str, script: str):
    """Process multiple files in single KFP component"""
    import subprocess
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all files in one component to amortize startup costs
    for i, input_file in enumerate(input_files):
        result = subprocess.run([
            'awk', script, input_file
        ], capture_output=True, text=True, check=True)
        
        output_file = output_path / f"result_{i}.txt"
        output_file.write_text(result.stdout)
    
    return str(output_path)
```

#### 3. **Container-Optimized AWK Scripts**

```python
def create_kfp_awk_component(script: str, component_name: str):
    """Create KFP component with embedded AWK script"""
    
    @kfp.components.create_component_from_func
    def awk_component(input_path: str, output_path: str) -> str:
        import subprocess
        from pathlib import Path
        
        # Write script to container filesystem
        script_path = Path("/tmp/awk_script.awk")
        script_path.write_text(script)
        
        # Execute AWK with script file
        result = subprocess.run([
            'awk', '-f', str(script_path), input_path
        ], capture_output=True, text=True, check=True)
        
        Path(output_path).write_text(result.stdout)
        return output_path
    
    awk_component.component_spec.name = component_name
    return awk_component
```

#### 4. **Memory-Efficient Streaming for Large Files**

```python
def kfp_stream_awk_processor(input_path: str, output_path: str, script: str):
    """Stream processing for large files in KFP"""
    import subprocess
    from pathlib import Path
    
    # Use streaming to handle large files within memory limits
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        process = subprocess.Popen([
            'awk', script
        ], stdin=infile, stdout=outfile, text=True)
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"AWK process failed with code {process.returncode}")
    
    return output_path
```

### KFP Pipeline Architecture

```python
@kfp.dsl.pipeline(
    name="receipt-processing-pipeline",
    description="Process receipts using AWK in KFP"
)
def receipt_processing_pipeline(
    input_dataset: str,
    fuel_extraction_script: str,
    bank_extraction_script: str
):
    # Component 1: Fuel receipt processing
    fuel_results = kfp_awk_processor(
        input_path=input_dataset,
        output_path="/tmp/fuel_results.txt",
        script=fuel_extraction_script
    )
    
    # Component 2: Bank statement processing  
    bank_results = kfp_awk_processor(
        input_path=fuel_results.output,
        output_path="/tmp/bank_results.txt", 
        script=bank_extraction_script
    )
    
    # Component 3: Results aggregation
    final_results = aggregate_results(
        fuel_results=fuel_results.output,
        bank_results=bank_results.output
    )
```

### KFP-Specific Optimizations

1. **Avoid Connection Pooling**: Not beneficial due to container lifecycle
2. **Minimize Process Creation**: Batch operations within single components
3. **Use Persistent Volumes**: For large datasets that don't fit in memory
4. **Resource Requests**: Specify appropriate CPU/memory limits
5. **Container Images**: Pre-install AWK and dependencies in base image

### Modified Recommendations for KFP

| Use Case | Standard Environment | KFP Environment |
|----------|---------------------|-----------------|
| **Small Files** | Connection Pooling | Stateless File Processing |
| **Large Files** | Memory-Mapped Files | Streaming + Persistent Volumes |
| **Batch Processing** | Async Processing | Single Component Batching |
| **Script Reuse** | Compiled Scripts | Container-Embedded Scripts |
| **Performance** | Persistent Processes | Resource-Optimized Components |

### Resource Configuration

```yaml
# KFP component resource specification
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi" 
    cpu: "1000m"
```

### Container Image Requirements

```dockerfile
FROM python:3.11-slim

# Install AWK and dependencies
RUN apt-get update && apt-get install -y \
    gawk \
    && rm -rf /var/lib/apt/lists/*

# Copy AWK scripts
COPY scripts/ /app/scripts/

WORKDIR /app
```

The KFP environment fundamentally changes the optimization strategy from persistent, long-running processes to efficient, stateless, containerized components that maximize throughput within resource constraints.