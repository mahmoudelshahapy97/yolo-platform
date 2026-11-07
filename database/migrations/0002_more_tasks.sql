-- Pose, instance segmentation and sequence (non-image) models join detect and classify.
-- Sequence models take a feature window instead of an image: their imgsz is 0 and
-- the input shape lives in the version's model_meta.json.
ALTER TABLE model_version DROP CONSTRAINT IF EXISTS model_version_task_check;
ALTER TABLE model_version ADD CONSTRAINT model_version_task_check
    CHECK (task IN ('detect', 'classify', 'pose', 'segment', 'sequence'));
