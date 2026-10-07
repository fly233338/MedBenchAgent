Understand the dataset once. Output only mapping.json and image_conversion_plan.json.
Mapping.images explicitly maps each final relative image locator to source_id, source_path, source_format and policy_id.
You choose locators under images/; retain JPEG/PNG extensions for unchanged ordinary images and use PNG for medical displays.
Reference-only mask locators identify features in Cases, not model input files. Document Case identity, joins, fields and missingness.
Each conversion policy records the required window, orientation, axis/slice or montage decisions using existing PresentationSpec fields.
Do not invent output digests/dimensions or claim conversion succeeded. Inspection scripts and observations stay in work/.
Ask only for missing semantic facts; preserve unknown vs negative. Submit your two decisions after validating their schemas and source references.
