# gaze_physics.py
import math

DEFAULT_GAZE_ANOMALY_THRESHOLD = 0.5

# --- VECTOR MATH HELPERS ---

def cross_product(v1, v2):
    """Calculates the orthogonal vector to v1 and v2."""
    return [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]

def dot_product(v1, v2):
    """Calculates the scalar dot product of v1 and v2."""
    return sum(x * y for x, y in zip(v1, v2))

def vector_magnitude(v):
    """Calculates the length of a vector."""
    return math.sqrt(sum(x**2 for x in v))

def vector_subtract(v1, v2):
    """Subtracts v2 from v1."""
    return [x - y for x, y in zip(v1, v2)]

def vector_add(v1, v2):
    """Adds two vectors."""
    return [x + y for x, y in zip(v1, v2)]

def scalar_multiply(scalar, v):
    """Multiplies a vector by a scalar."""
    return [scalar * x for x in v]

def normalize_vector(v):
    """Returns a unit-length vector with the same direction as v."""
    magnitude = vector_magnitude(v)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return [x / magnitude for x in v]

def _validate_coordinate_vector(name, vec):
    """Validate that vec is a 3-element numeric coordinate vector."""
    if not isinstance(vec, (list, tuple)):
        raise TypeError(f"'{name}' must be a list or tuple. Received: {type(vec).__name__}")
    if len(vec) != 3:
        raise ValueError(f"'{name}' must be a 3-element list or tuple. Received length {len(vec)}.")
    if not all(isinstance(val, (int, float)) for val in vec):
        raise TypeError(f"All elements in '{name}' must be numeric (int or float).")
    return [float(val) for val in vec]

# --- CORE PHYSICS ENGINE ---

def calculate_vergence_point(p_l, p_r, g_l, g_r):
    """
    Finds the 3D midpoint of the shortest line segment connecting the two gaze rays.
    This acts as our approximated 3D vergence point (where the eyes are looking).

    Args:
        p_l: Left pupil position as a 3D point.
        p_r: Right pupil position as a 3D point.
        g_l: Left eye gaze direction vector.
        g_r: Right eye gaze direction vector.

    Returns:
        None if the gaze rays are parallel or nearly parallel; otherwise a tuple
        of (vergence_point, t_l, t_r), where vergence_point is the midpoint of
        the shortest segment between the rays and t_l and t_r are the line
        parameters for the closest points on the left and right gaze rays.
    """
    w = vector_subtract(p_l, p_r)

    g_l_squared = dot_product(g_l, g_l)
    g_l_dot_g_r = dot_product(g_l, g_r)
    g_r_squared = dot_product(g_r, g_r)
    g_l_dot_w = dot_product(g_l, w)
    g_r_dot_w = dot_product(g_r, w)

    denominator = g_l_squared * g_r_squared - g_l_dot_g_r * g_l_dot_g_r
    epsilon = 1e-10

    # If denominator is effectively 0, lines are parallel or nearly parallel
    if abs(denominator) < epsilon:
        return None

    # Calculate the parameters for the closest points on each line
    t_l = (g_l_dot_g_r * g_r_dot_w - g_r_squared * g_l_dot_w) / denominator
    t_r = (g_l_squared * g_r_dot_w - g_l_dot_g_r * g_l_dot_w) / denominator

    # Find the actual closest 3D points on the left and right gaze rays
    closest_point_left = vector_add(p_l, scalar_multiply(t_l, g_l))
    closest_point_right = vector_add(p_r, scalar_multiply(t_r, g_r))

    # The vergence point is the exact midpoint between these two closest points
    vergence_point = scalar_multiply(0.5, vector_add(closest_point_left, closest_point_right))

    return vergence_point, t_l, t_r

def calculate_gaze_anomaly(p_l, p_r, g_l, g_r):
    """
    Return the shortest distance between the two gaze rays.

    Args:
        p_l (list[float] | tuple[float, float, float]): 3D position of the
            left pupil, used as the origin of the left gaze ray.
        p_r (list[float] | tuple[float, float, float]): 3D position of the
            right pupil, used as the origin of the right gaze ray.
        g_l (list[float] | tuple[float, float, float]): 3D gaze direction
            vector for the left eye.
        g_r (list[float] | tuple[float, float, float]): 3D gaze direction
            vector for the right eye.

    Returns:
        float: The shortest distance between the two gaze rays. Returns
        float('inf') if the rays are parallel or nearly parallel and do not
        produce a finite closest-approach distance.
    """
    # 1. Find the orthogonal normal vector
    n = cross_product(g_l, g_r)
    
    # Check if lines are perfectly parallel using an epsilon threshold
    mag_n = vector_magnitude(n)
    if mag_n < 1e-10:
        return float('inf') # Parallel lines that don't converge
        
    # 2. Vector connecting the pupils
    p_diff = vector_subtract(p_r, p_l)

    # 3. Orthogonal projection to find the shortest distance
    distance = abs(dot_product(p_diff, n)) / mag_n

    return distance

def is_deepfake_gaze(distance, threshold=DEFAULT_GAZE_ANOMALY_THRESHOLD):
    """Classify a gaze anomaly distance against the configured threshold."""
    return distance > threshold

def analyze_gaze(p_l, p_r, g_l, g_r, threshold=DEFAULT_GAZE_ANOMALY_THRESHOLD):
    """
    Analyze binocular gaze geometry and classify whether it appears anomalous.

    Args:
        p_l: Left pupil position as a 3D point/vector representing the origin of
            the left gaze ray.
        p_r: Right pupil position as a 3D point/vector representing the origin of
            the right gaze ray.
        g_l: Left eye gaze direction vector.
        g_r: Right eye gaze direction vector.
        threshold: Distance threshold used to classify the gaze geometry as a
            deepfake anomaly. Distances greater than this value are classified as
            anomalous.

    Returns:
        A dictionary with the following keys:
            "distance": The shortest distance between the two gaze rays. May be
                `float('inf')` when the geometry is illogical or the rays are
                effectively parallel.
            "threshold": The threshold value used for classification.
            "is_deepfake": `True` if the gaze is classified as anomalous/deepfake,
                otherwise `False`.
            "vergence_point": The computed 3D vergence/intersection point if one
                is available, otherwise `None`.
            "logical_gaze": `True` when the gaze geometry is physically plausible
                (for example, the eyes converge in front of the viewer), otherwise
                `False`.
    """
    distance = calculate_gaze_anomaly(p_l, p_r, g_l, g_r)
    
    # Calculate where the eyes are actually looking
    vergence_data = calculate_vergence_point(p_l, p_r, g_l, g_r)
    
    vergence_point = None
    logical_gaze = True
    
    if vergence_data:
        vergence_point, t_l, t_r = vergence_data
        vergence_point = list(vergence_point)

        # GEOMETRIC ATTESTATION: Are the eyes looking backwards or in opposite directions?
        # If t_l or t_r is negative, the intersection is BEHIND the eyes.
        if t_l < 0 or t_r < 0:
            logical_gaze = False
            distance = float('inf')
    else:
        # Parallel gaze rays do not converge, which is geometrically illogical
        # for human binocular vision.
        logical_gaze = False
        distance = float('inf')
            
    is_fake = is_deepfake_gaze(distance, threshold) or not logical_gaze

    return {
        "distance": distance,
        "threshold": threshold,
        "is_deepfake": is_fake,
        "vergence_point": vergence_point,
        "logical_gaze": logical_gaze
    }

# --- API WRAPPER ---

def process_api_payload(payload):
    """
    Process a JSON-like payload containing binocular gaze geometry inputs.

    Args:
        payload (dict): Input mapping with the following structure:
            Required keys:
                - "left_pupil": 3-element list/tuple of numeric coordinates [x, y, z]
                - "right_pupil": 3-element list/tuple of numeric coordinates [x, y, z]
                - "left_gaze": 3-element list/tuple representing the left eye gaze vector
                - "right_gaze": 3-element list/tuple representing the right eye gaze vector
            Optional keys:
                - "threshold": non-negative int/float used to classify the anomaly distance;
                  defaults to DEFAULT_GAZE_ANOMALY_THRESHOLD when omitted

    Returns:
        dict: On success, returns the result of ``analyze_gaze(...)`` with keys:
            - "distance" (float): computed gaze anomaly distance
            - "threshold" (int|float): threshold used for classification
            - "is_deepfake" (bool): whether the gaze is classified as anomalous/fake
            - "vergence_point" (list|tuple|None): estimated intersection point of gaze rays,
              or None if no vergence point is found
            - "logical_gaze" (bool): whether the gaze geometry is physically plausible

        dict: On error, returns:
            - {"error": "Missing required coordinate data: ..."} when a required key is absent
            - {"error": "Data Validation Error: ..."} when values fail type/shape validation
            - {"error": "Unexpected error processing payload: ..."} for any other exception
    """
    try:
        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dictionary with gaze geometry fields.")

        p_l = _validate_coordinate_vector("left_pupil", payload["left_pupil"])
        p_r = _validate_coordinate_vector("right_pupil", payload["right_pupil"])
        g_l = _validate_coordinate_vector("left_gaze", payload["left_gaze"])
        g_r = _validate_coordinate_vector("right_gaze", payload["right_gaze"])
        
        # Extract and validate optional threshold
        threshold = payload.get("threshold", DEFAULT_GAZE_ANOMALY_THRESHOLD)
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("Optional 'threshold' must be a non-negative number.")

        # Ensure the gaze directions are valid non-zero vectors
        if vector_magnitude(g_l) == 0 or vector_magnitude(g_r) == 0:
            raise ValueError("Gaze direction vectors must be non-zero.")

        return analyze_gaze(p_l, p_r, g_l, g_r, threshold=threshold)

    except KeyError as e:
        return {"error": f"Missing required coordinate data: {str(e)}"}
    except (ValueError, TypeError) as e:
        return {"error": f"Data Validation Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error processing payload: {str(e)}"}

# --- SANITY CHECK TESTS ---

if __name__ == "__main__":
    print("Testing Gaze Physics Engine...\n")

    # TEST CASE 1: REAL GAZE (Perfectly Converging)
    real_payload = {
        "left_pupil": [-1, 0, 0],
        "right_pupil": [1, 0, 0],
        "left_gaze": [1, 0, 10],
        "right_gaze": [-1, 0, 10]
    }
    real_result = process_api_payload(real_payload)
    print(f"1. Real Gaze Test     -> Fake: {real_result.get('is_deepfake')} | Logical: {real_result.get('logical_gaze')}")

    # TEST CASE 2: FAKE GAZE (Skewed Lines)
    fake_payload = {
        "left_pupil": [-1, 0, 0],
        "right_pupil": [1, 0, 0],
        "left_gaze": [1, 1, 10],
        "right_gaze": [-1, -2, 10]
    }
    fake_result = process_api_payload(fake_payload)
    print(f"2. Fake Gaze Test     -> Fake: {fake_result.get('is_deepfake')} | Logical: {fake_result.get('logical_gaze')}")

    # TEST CASE 3: VALIDATION FAILURE (Bad Data)
    bad_payload = {
        "left_pupil": [-1, 0], # Only 2 dimensions
        "right_pupil": [1, 0, 0],
        "left_gaze": [1, 1, "ten"], # String instead of int
        "right_gaze": [-1, -2, 10],
        "threshold": -5 # Invalid negative threshold
    }
    bad_result = process_api_payload(bad_payload)
    print(f"3. Bad Payload Test   -> Result: {bad_result}")
    
    # TEST CASE 4: BACKWARD GAZE (Geometric Attestation Failure)
    backward_payload = {
        "left_pupil": [-1, 0, 0],
        "right_pupil": [1, 0, 0],
        "left_gaze": [1, 0, -10], # Looking backwards behind the face
        "right_gaze": [-1, 0, -10]
    }
    backward_result = process_api_payload(backward_payload)
    print(f"4. Backward Gaze Test -> Fake: {backward_result.get('is_deepfake')} | Logical: {backward_result.get('logical_gaze')}")

    # TEST CASE 5: PARALLEL GAZE
    parallel_payload = {
        "left_pupil": [-1, 0, 0],
        "right_pupil": [1, 0, 0],
        "left_gaze": [0, 0, 10],
        "right_gaze": [0, 0, 10]
    }
    parallel_result = process_api_payload(parallel_payload)
    print(f"5. Parallel Gaze Test -> Fake: {parallel_result.get('is_deepfake')} | Logical: {parallel_result.get('logical_gaze')}")