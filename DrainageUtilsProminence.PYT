# -*- coding: utf-8 -*-

import arcpy
import os
import collections
import math
import gc # Garbage Collector for memory management

import numpy as np
from scipy.signal import find_peaks

# Define the toolbox class
class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the .pyt file)."""
        self.label = "Drainage Utilities Prominence"
        self.alias = "DrainageUtilsProminence"

        # List of tool classes associated with this toolbox
        self.tools = [FindExtremaProminence]

# Define the tool class
class FindExtremaProminence(object):
    def __init__(self):
        """Define the tool class."""
        self.label = "Find Extrema From Lines (Prominence)"
        self.description = ("Generates points along input lines, extracts Z from DEM, identifies absolute and significant "
                            "local elevation extrema using prominence, applies end exclusion, minimum distance, and alternating "
                            "Min/Max type filtering (prioritizing Absolute points), and outputs final extrema points.")
        self.canRunInBackground = False
        self.category = "Drainage Analysis"

    def getParameterInfo(self):
        """Define parameter definitions"""

        # Parameter 0: Input Line Features
        param_in_lines = arcpy.Parameter(
            displayName="Input Line Features",
            name="in_line_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        param_in_lines.filter.list = ["Polyline"]
        param_in_lines.description = "The input polyline features (e.g., curbs, ditches) for which to find extrema points."

        # Parameter 1: Route Identifier Field (from Input Lines)
        param_route_id_field = arcpy.Parameter(
            displayName="Route Identifier Field (from Input Lines)",
            name="route_id_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        param_route_id_field.parameterDependencies = [param_in_lines.name]
        param_route_id_field.filter.list = ["Short", "Long", "OID", "Guid", "String", "Integer"]
        param_route_id_field.description = "Field in the Input Line Features containing a unique ID for each line (e.g., OBJECTID, Curb_Route_ID). This ID will be transferred to the output points."

        # Parameter 2: Input DEM Raster
        param_in_dem = arcpy.Parameter(
            displayName="Input DEM Raster (Conditioned)",
            name="in_dem_raster",
            datatype="GPRasterLayer", # GPRasterLayer allows getting catalogPath
            parameterType="Required",
            direction="Input")
        param_in_dem.description = "The conditioned Digital Elevation Model raster used to extract elevation (Z) values for points along the lines."

        # Parameter 3: Output Extrema Points Feature Class
        param_out_points = arcpy.Parameter(
            displayName="Output Extrema Points Feature Class",
            name="out_extrema_points",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output")
        param_out_points.description = "The output point feature class where the identified and filtered extrema points will be saved."

        # --- Analysis Parameters ---
        large_upper_bound = 1e9

        # Parameter 4: Sampling Interval (ft)
        param_sampling_interval = arcpy.Parameter(
            displayName="Sampling Interval Along Lines (ft)",
            name="sampling_interval",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_sampling_interval.value = 4.0
        param_sampling_interval.filter.type = "Range"
        param_sampling_interval.filter.list = [0.1, large_upper_bound]
        param_sampling_interval.description = "The interval (in feet) at which points will be generated along the input lines for elevation sampling (e.g., 4.0)."

        # Parameter 5: Prominence (ft)
        param_prominence = arcpy.Parameter(
            displayName="Minimum Prominence (ft)",
            name="prominence",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_prominence.value = 0.5
        param_prominence.filter.type = "Range"
        param_prominence.filter.list = [0, large_upper_bound]
        param_prominence.description = "The minimum vertical distance (in feet) a local peak or valley must 'stand out' from its surroundings to be considered significant (e.g., 0.5)."

        # Parameter 6: Minimum Horizontal Distance (ft)
        param_min_distance = arcpy.Parameter(
            displayName="Minimum Horizontal Distance Between Final Extrema (ft)",
            name="min_horizontal_distance",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_min_distance.value = 50.0
        param_min_distance.filter.type = "Range"
        param_min_distance.filter.list = [0, large_upper_bound]
        param_min_distance.description = "The minimum horizontal distance (in feet, measured along the line) allowed between the FINAL selected alternating extrema points."

        # Parameter 7: End Exclusion Distance (ft)
        param_end_exclusion = arcpy.Parameter(
            displayName="End Exclusion Distance (ft)",
            name="end_exclusion_distance",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        param_end_exclusion.value = 30.0
        param_end_exclusion.filter.type = "Range"
        param_end_exclusion.filter.list = [0, large_upper_bound]
        param_end_exclusion.description = "The distance (in feet) from the start and end of each input line within which identified extrema points will be ignored (e.g., 30.0)."


        params = [param_in_lines, param_route_id_field, param_in_dem,
                  param_out_points, param_sampling_interval, param_prominence,
                  param_min_distance, param_end_exclusion]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        # Requires Spatial Analyst
        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                raise Exception("Spatial Analyst license is unavailable.")
            return True
        except Exception as e:
            self.isLicensed = lambda: False
            arcpy.AddError(str(e))
            return False

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""
        if parameters[0].valueAsText and not parameters[3].altered:
            try:
                in_lines_desc = arcpy.Describe(parameters[0].valueAsText)
                out_ws = os.path.dirname(in_lines_desc.catalogPath) if in_lines_desc.catalogPath else arcpy.env.scratchGDB
                out_name = arcpy.ValidateTableName(f"{in_lines_desc.baseName}_ExtremaPtsProminence", out_ws)
                parameters[3].value = os.path.join(out_ws, out_name)
            except:
                 parameters[3].value = os.path.join(arcpy.env.scratchGDB, "ExtremaPtsProminence")
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation."""
        line_fc_idx = 0
        route_id_idx = 1
        sampling_idx = 4
        prominence_idx = 5
        min_dist_idx = 6
        end_excl_idx = 7

        if parameters[sampling_idx].value is not None and parameters[sampling_idx].value <= 0:
             parameters[sampling_idx].setErrorMessage("Sampling interval must be positive.")
        if parameters[prominence_idx].value is not None and parameters[prominence_idx].value < 0:
            parameters[prominence_idx].setErrorMessage("Prominence must be zero or positive.")
        if parameters[min_dist_idx].value is not None and parameters[min_dist_idx].value < 0:
            parameters[min_dist_idx].setErrorMessage("Minimum horizontal distance must be zero or positive.")
        if parameters[end_excl_idx].value is not None and parameters[end_excl_idx].value < 0:
            parameters[end_excl_idx].setErrorMessage("End exclusion distance must be zero or positive.")

        if parameters[line_fc_idx].value and parameters[route_id_idx].value:
            try:
                field_names = [f.name.lower() for f in arcpy.ListFields(parameters[line_fc_idx].valueAsText)]
                oid_field_name = arcpy.Describe(parameters[line_fc_idx].valueAsText).OIDFieldName.lower()
                if parameters[route_id_idx].valueAsText.lower() not in field_names and \
                   parameters[route_id_idx].valueAsText.lower() != oid_field_name:
                    parameters[route_id_idx].setErrorMessage(f"Route ID Field '{parameters[route_id_idx].valueAsText}' not found in {parameters[line_fc_idx].valueAsText}.")
            except Exception as e:
                 arcpy.AddWarning(f"Could not verify Route ID field: {e}")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        in_lines_fc = parameters[0].valueAsText
        route_id_field_input_name = parameters[1].valueAsText
        in_dem_param = parameters[2] # Get the parameter object itself
        in_dem_raster_path = in_dem_param.valueAsText # Get path/layer name provided by user
        out_points_fc = parameters[3].valueAsText
        sampling_interval = parameters[4].value
        prominence_param = parameters[5].value
        min_distance_ft = parameters[6].value
        end_exclusion_dist = parameters[7].value

        arcpy.AddMessage(f"Input Lines: {in_lines_fc}")
        arcpy.AddMessage(f"Route ID Field from Input: {route_id_field_input_name}")
        arcpy.AddMessage(f"Input DEM Parameter Value: {in_dem_raster_path}")
        arcpy.AddMessage(f"Output Points: {out_points_fc}")
        arcpy.AddMessage(f"Sampling Interval: {sampling_interval} ft")
        arcpy.AddMessage(f"Prominence: {prominence_param} ft")
        arcpy.AddMessage(f"Min Horizontal Distance (Final Points): {min_distance_ft} ft")
        arcpy.AddMessage(f"End Exclusion Distance: {end_exclusion_dist} ft")

        arcpy.env.overwriteOutput = True
        out_ws = os.path.dirname(out_points_fc)
        if not arcpy.Exists(out_ws):
             raise ValueError(f"Output workspace does not exist: {out_ws}")

        temp_points_basename = arcpy.ValidateTableName("temp_line_points_input", out_ws)
        temp_points_input_path = os.path.join(out_ws, temp_points_basename)
        temp_points_extracted_basename = arcpy.ValidateTableName("temp_line_points_extracted", out_ws)
        temp_points_extracted_path = os.path.join(out_ws, temp_points_extracted_basename)
        arcpy.AddMessage(f"Using temporary input points path: {temp_points_input_path}")
        arcpy.AddMessage(f"Using temporary extracted points path: {temp_points_extracted_path}")

        arcpy.CheckOutExtension("Spatial")

        # --- Helper function to classify point type ---
        def get_general_type(point_type_string):
            if point_type_string in ['AbsMax', 'LocalMax']: return 'Max'
            elif point_type_string in ['AbsMin', 'LocalMin']: return 'Min'
            else: return None
        # --- End helper ---

        # Store original environment settings to restore later
        orig_cell_size = arcpy.env.cellSize
        orig_extent = arcpy.env.extent
        orig_mask = arcpy.env.mask

        try:
            arcpy.AddMessage("Preparing data...")
            line_desc = arcpy.Describe(in_lines_fc)
            sr = line_desc.spatialReference
            input_oid_field_name = line_desc.OIDFieldName
            if route_id_field_input_name.upper() == input_oid_field_name.upper():
                 route_id_field_to_read = input_oid_field_name
            else:
                 route_id_field_to_read = route_id_field_input_name

            # --- Use Describe to get the full catalog path of the DEM ---
            dem_desc = arcpy.Describe(in_dem_raster_path)
            dem_catalog_path = dem_desc.catalogPath
            arcpy.AddMessage(f"Resolved DEM Catalog Path: {dem_catalog_path}")
            if not arcpy.Exists(dem_catalog_path):
                 raise ValueError(f"Input DEM raster not found at resolved path: {dem_catalog_path}")
            # --- END ---

            # --- REMOVED GetRasterProperties calls and explicit env setting ---

            arcpy.AddMessage(f"Creating output feature class: {out_points_fc}")
            out_path, out_name = os.path.split(out_points_fc)
            if arcpy.Exists(out_points_fc): arcpy.management.Delete(out_points_fc)
            arcpy.management.CreateFeatureclass(out_path, out_name, "POINT", spatial_reference=sr)

            output_route_id_field_name = "Source_Line_OID"
            if route_id_field_input_name.upper() not in ["OBJECTID", "FID"]:
                 output_route_id_field_name = route_id_field_input_name

            input_fields_list = arcpy.ListFields(in_lines_fc, route_id_field_to_read)
            if not input_fields_list: raise ValueError(f"Route ID field '{route_id_field_to_read}' not found.")
            input_field_obj = input_fields_list[0]
            input_field_type = input_field_obj.type.upper()

            if input_field_type in ["OID", "INTEGER"]: output_id_type = "LONG"
            elif input_field_type == "GUID": output_id_type = "GUID"
            elif input_field_type == "DOUBLE": output_id_type = "DOUBLE"
            elif input_field_type == "SINGLE": output_id_type = "FLOAT"
            elif input_field_type == "DATE": output_id_type = "DATE"
            else:
                output_id_type = "TEXT"
                output_id_length = input_field_obj.length
                arcpy.AddMessage(f"Adding Route ID field '{output_route_id_field_name}' as type {output_id_type}({output_id_length}).")
                arcpy.management.AddField(out_points_fc, output_route_id_field_name, output_id_type, field_length=output_id_length)

            if output_id_type != "TEXT":
                 arcpy.AddMessage(f"Adding Route ID field '{output_route_id_field_name}' as type {output_id_type}.")
                 arcpy.management.AddField(out_points_fc, output_route_id_field_name, output_id_type)

            arcpy.management.AddField(out_points_fc, "PointType", "TEXT", field_length=10)
            arcpy.management.AddField(out_points_fc, "Elevation", "DOUBLE")
            arcpy.management.AddField(out_points_fc, "M_Value", "DOUBLE")
            arcpy.management.AddField(out_points_fc, "Prominence", "DOUBLE")

            arcpy.AddMessage("Processing lines...")
            line_fields = ["SHAPE@", route_id_field_to_read, "OID@", "SHAPE@LENGTH"]
            final_extrema_to_write = []
            processed_line_count = 0

            total_lines = int(arcpy.management.GetCount(in_lines_fc)[0])
            arcpy.SetProgressor("step", "Processing lines...", 0, total_lines, 1)

            with arcpy.da.SearchCursor(in_lines_fc, line_fields) as cursor:
                for row in cursor:
                    processed_line_count += 1
                    line_geom, route_id, line_oid, line_length = row
                    arcpy.SetProgressorLabel(f"Processing Line {processed_line_count}/{total_lines} (ID: {route_id})...")

                    if line_geom is None or line_length is None or line_length < (2 * sampling_interval):
                        messages.addWarningMessage(f"Skipping Route ID {route_id} (In OID: {line_oid}): Null geometry or too short.")
                        arcpy.SetProgressorPosition(processed_line_count)
                        continue

                    generated_points_data = []
                    distance = 0.0
                    while distance <= line_length:
                        try:
                            point = line_geom.positionAlongLine(distance)
                            if point is None:
                                messages.addWarningMessage(f"Failed point gen at dist {distance:.2f}, RID {route_id}.")
                                distance += sampling_interval
                                continue
                            generated_points_data.append({'geom': arcpy.PointGeometry(point.firstPoint, sr), 'm': distance})
                        except Exception as pos_ex:
                             messages.addWarningMessage(f"Error point gen at dist {distance:.2f}, RID {route_id}: {pos_ex}.")

                        distance += sampling_interval
                        if distance > line_length and (distance - sampling_interval) < line_length:
                             try:
                                 end_point = line_geom.positionAlongLine(line_length)
                                 if end_point: generated_points_data.append({'geom': arcpy.PointGeometry(end_point.firstPoint, sr), 'm': line_length})
                             except Exception as end_pos_ex: messages.addWarningMessage(f"Error end point RID {route_id}: {end_pos_ex}.")

                    if len(generated_points_data) < 3:
                        arcpy.SetProgressorPosition(processed_line_count)
                        continue

                    points_with_z = []
                    try:
                        # --- REMOVED explicit cell size setting ---

                        if arcpy.Exists(temp_points_input_path): arcpy.management.Delete(temp_points_input_path)
                        if arcpy.Exists(temp_points_extracted_path): arcpy.management.Delete(temp_points_extracted_path)

                        point_geometries = [pt['geom'] for pt in generated_points_data]
                        arcpy.management.CopyFeatures(point_geometries, temp_points_input_path)

                        arcpy.management.AddField(temp_points_input_path, "M_Value", "DOUBLE")
                        with arcpy.da.UpdateCursor(temp_points_input_path, ["M_Value"], sql_clause=(None, "ORDER BY OBJECTID")) as u_cursor:
                            m_index = 0
                            for u_row in u_cursor:
                                if m_index < len(generated_points_data):
                                    u_row[0] = generated_points_data[m_index]['m']
                                    u_cursor.updateRow(u_row)
                                m_index += 1

                        # Use DEM catalog path
                        arcpy.sa.ExtractValuesToPoints(temp_points_input_path, dem_catalog_path, temp_points_extracted_path,"NONE", "VALUE_ONLY")

                        extract_field = "RASTERVALU"
                        temp_fields = [f.name for f in arcpy.ListFields(temp_points_extracted_path)]
                        if extract_field not in temp_fields:
                             raise ValueError(f"'{extract_field}' field not found in {temp_points_extracted_path} after ExtractValuesToPoints.")

                        with arcpy.da.SearchCursor(temp_points_extracted_path, ["SHAPE@", extract_field, "M_Value", "OID@"]) as read_cursor:
                             for pt_row in read_cursor:
                                 shape, z_val, m_val, pt_oid = pt_row
                                 if z_val is None or z_val == -9999: continue
                                 points_with_z.append({'geom': shape, 'elev': z_val, 'm': m_val, 'orig_idx': len(points_with_z)})

                        arcpy.management.Delete(temp_points_input_path)
                        arcpy.management.Delete(temp_points_extracted_path)

                    except arcpy.ExecuteError as exec_err:
                        messages.addWarningMessage(f"ExtractValuesToPoints failed RID {route_id}: {arcpy.GetMessages(2)}. Skipping line.")
                        # --- TEMPORARILY COMMENT OUT DELETES for specific ID ---
                        # if route_id == 1: # Or another failing ID
                        #    messages.addWarningMessage(f"Keeping temp files for RID {route_id} for manual testing.")
                        # else:
                        #    if arcpy.Exists(temp_points_input_path): arcpy.management.Delete(temp_points_input_path)
                        #    if arcpy.Exists(temp_points_extracted_path): arcpy.management.Delete(temp_points_extracted_path)
                        # --- END TEMP COMMENT OUT ---
                        arcpy.SetProgressorPosition(processed_line_count)
                        continue # Skip to next line
                    except Exception as e_inner:
                        messages.addWarningMessage(f"Error processing points (Z extraction) RID {route_id}: {e_inner}. Skipping line.")
                        if arcpy.Exists(temp_points_input_path): arcpy.management.Delete(temp_points_input_path)
                        if arcpy.Exists(temp_points_extracted_path): arcpy.management.Delete(temp_points_extracted_path)
                        arcpy.SetProgressorPosition(processed_line_count)
                        continue # Skip to next line
                    # --- REMOVED finally block that reset cell size ---


                    if len(points_with_z) < 3:
                        arcpy.SetProgressorPosition(processed_line_count)
                        continue

                    # --- Find Extrema using SciPy ---
                    elevations = np.array([p['elev'] for p in points_with_z])
                    candidate_extrema_by_index = {}

                    abs_min_idx = np.argmin(elevations)
                    abs_max_idx = np.argmax(elevations)
                    candidate_extrema_by_index[abs_min_idx] = ('AbsMin', None)
                    if abs_min_idx != abs_max_idx: candidate_extrema_by_index[abs_max_idx] = ('AbsMax', None)

                    try: # Maxima (No distance filter here)
                        peak_indices, peak_props = find_peaks(elevations, prominence=prominence_param)
                        peak_proms = peak_props.get('prominences', [None]*len(peak_indices))
                        for i, idx in enumerate(peak_indices):
                             current_type, current_prom = candidate_extrema_by_index.get(idx, (None, -1))
                             if current_type not in ['AbsMin', 'AbsMax']:
                                  if peak_proms[i] is not None and peak_proms[i] > (current_prom if current_prom is not None else -1): candidate_extrema_by_index[idx] = ('LocalMax', peak_proms[i])
                             elif current_type is None: candidate_extrema_by_index[idx] = ('LocalMax', peak_proms[i])
                    except Exception as e: messages.addWarningMessage(f"Find peaks (max) failed RID {route_id}: {e}")

                    try: # Minima (No distance filter here)
                        valley_indices, valley_props = find_peaks(-elevations, prominence=prominence_param)
                        valley_proms = valley_props.get('prominences', [None]*len(valley_indices))
                        for i, idx in enumerate(valley_indices):
                             current_type, current_prom = candidate_extrema_by_index.get(idx, (None, -1))
                             if current_type not in ['AbsMin', 'AbsMax']:
                                   if valley_proms[i] is not None and valley_proms[i] > (current_prom if current_prom is not None else -1): candidate_extrema_by_index[idx] = ('LocalMin', valley_proms[i])
                             elif current_type is None: candidate_extrema_by_index[idx] = ('LocalMin', valley_proms[i])
                    except Exception as e: messages.addWarningMessage(f"Find peaks (min) failed RID {route_id}: {e}")

                    # --- Filter ALL candidates ---
                    candidates_for_final_filter = []
                    for idx, (pt_type, prominence_val) in candidate_extrema_by_index.items():
                        if idx < 0 or idx >= len(points_with_z): continue
                        point_details = points_with_z[idx]
                        m_value = point_details['m']
                        is_excl = False
                        if m_value is None: is_excl = True
                        else:
                            dist_from_end = max(0, line_length - m_value)
                            if m_value < (end_exclusion_dist - 1e-6) or dist_from_end < (end_exclusion_dist - 1e-6): is_excl = True
                        if not is_excl:
                            candidates_for_final_filter.append({'m': m_value, 'idx': idx, 'type': pt_type, 'prom': prominence_val, 'details': point_details})

                    candidates_for_final_filter.sort(key=lambda x: x['m'])

                    final_kept_points_this_line = []
                    for current_candidate in candidates_for_final_filter:
                        if not final_kept_points_this_line:
                            final_kept_points_this_line.append(current_candidate)
                            continue
                        last_kept_point = final_kept_points_this_line[-1]
                        distance_ok = current_candidate['m'] >= last_kept_point['m'] + min_distance_ft
                        current_general_type = get_general_type(current_candidate['type'])
                        last_general_type = get_general_type(last_kept_point['type'])
                        type_ok = current_general_type != last_general_type

                        if distance_ok and type_ok:
                            final_kept_points_this_line.append(current_candidate)
                        elif not type_ok: # Same type conflict
                            is_current_abs = current_candidate['type'] in ['AbsMin', 'AbsMax']
                            is_last_local = last_kept_point['type'] in ['LocalMin', 'LocalMax']
                            if is_current_abs and is_last_local:
                                messages.addMessage(f"RID {route_id}: Prioritizing {current_candidate['type']} at M={current_candidate['m']:.1f} over {last_kept_point['type']} at M={last_kept_point['m']:.1f}")
                                final_kept_points_this_line[-1] = current_candidate

                    for kept_point_data in final_kept_points_this_line:
                        final_extrema_to_write.append((
                             route_id, kept_point_data['type'], kept_point_data['details']['elev'],
                             kept_point_data['m'], kept_point_data['prom'], kept_point_data['details']['geom']
                         ))

                    arcpy.SetProgressorPosition(processed_line_count)
                    # if processed_line_count % 20 == 0: gc.collect()

            arcpy.ResetProgressor()

            # --- Write Output Points ---
            num_found = len(final_extrema_to_write)
            arcpy.AddMessage(f"Identified {num_found} points after all filters. Writing output...")
            if not final_extrema_to_write:
                messages.addWarningMessage("No extrema found matching all criteria.")
                return

            output_fields = [output_route_id_field_name, "PointType", "Elevation", "M_Value", "Prominence", "SHAPE@"]
            with arcpy.da.InsertCursor(out_points_fc, output_fields) as insert_cursor:
                for row_data in final_extrema_to_write: insert_cursor.insertRow(row_data)
            arcpy.AddMessage(f"Successfully wrote {num_found} points.")

        except ImportError as ie: messages.addError(f"Import Error: {ie}. Ensure NumPy & SciPy installed.")
        except arcpy.ExecuteError:
             # --- FIX for error logging ---
             messages.addError(f"ArcPy Execute Error: {arcpy.GetMessages(2)}")
             # --- END FIX ---
        except Exception as e:
            # --- FIX for error logging ---
            messages.addError(f"An unexpected error occurred: {str(e)}")
            # --- END FIX ---
            import traceback
            arcpy.AddError(traceback.format_exc())
        finally:
            # --- Reset environments and clean up ---
            arcpy.env.cellSize = orig_cell_size
            arcpy.env.extent = orig_extent
            arcpy.env.mask = orig_mask
            if arcpy.Exists(temp_points_input_path): arcpy.management.Delete(temp_points_input_path)
            if arcpy.Exists(temp_points_extracted_path): arcpy.management.Delete(temp_points_extracted_path)
            arcpy.CheckInExtension("Spatial")
            # --- END ---

        return

    def postExecute(self, parameters):
        """Runs after execution."""
        arcpy.AddMessage("Extrema identification (Prominence) complete.")
        return

