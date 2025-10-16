import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any
import json

class XMLShapeParser:
    """Parser for extracting all shape annotations from XML files."""
    
    def __init__(self, xml_file_path: str):
        """Initialize parser with XML file path."""
        self.xml_file_path = xml_file_path
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()
        
    def get_labels(self) -> set:
        """Extract all unique labels/classes from the XML."""
        labels = set()
        for label_elem in self.root.findall(".//labels/label"):
            labels.add(label_elem.find("name").text)
        return labels
    
    def parse_ellipses(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all ellipse annotations organized by frame."""
        ellipses_per_frame = {}
        
        for ellipse in self.root.findall(".//ellipse"):
            frame = int(ellipse.get("frame"))
            if frame not in ellipses_per_frame:
                ellipses_per_frame[frame] = []
                
            # Get the track label from parent track element
            # Find the parent track element
            label = "unknown"
            for track in self.root.findall(".//track"):
                if ellipse in track:
                    label = track.get("label", "unknown")
                    break
            
            ellipse_data = {
                "shape_type": "ellipse",
                "label": label,
                "frame": frame,
                "cx": float(ellipse.get("cx")),  # center x
                "cy": float(ellipse.get("cy")),  # center y
                "rx": float(ellipse.get("rx")),  # radius x
                "ry": float(ellipse.get("ry")),  # radius y
                "keyframe": bool(int(ellipse.get("keyframe", "0"))),
                "outside": bool(int(ellipse.get("outside", "0"))),
                "occluded": bool(int(ellipse.get("occluded", "0"))),
                "z_order": int(ellipse.get("z_order", "0"))
            }
            ellipses_per_frame[frame].append(ellipse_data)
            
        return ellipses_per_frame
    
    def parse_boxes(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all box annotations organized by frame."""
        boxes_per_frame = {}
        
        for box in self.root.findall(".//box"):
            frame = int(box.get("frame"))
            if frame not in boxes_per_frame:
                boxes_per_frame[frame] = []
                
            box_data = {
                "shape_type": "box",
                "label": box.get("label"),
                "frame": frame,
                "xtl": float(box.get("xtl")),  # top-left x
                "ytl": float(box.get("ytl")),  # top-left y
                "xbr": float(box.get("xbr")),  # bottom-right x
                "ybr": float(box.get("ybr")),  # bottom-right y
                "keyframe": bool(int(box.get("keyframe", "0"))),
                "outside": bool(int(box.get("outside", "0"))),
                "occluded": bool(int(box.get("occluded", "0"))),
                "z_order": int(box.get("z_order", "0"))
            }
            boxes_per_frame[frame].append(box_data)
            
        return boxes_per_frame
    
    def parse_polygons(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all polygon annotations organized by frame."""
        polygons_per_frame = {}
        
        for polygon in self.root.findall(".//polygon"):
            frame = int(polygon.get("frame"))
            if frame not in polygons_per_frame:
                polygons_per_frame[frame] = []
                
            # Get the track label from parent track element
            # Find the parent track element
            label = "unknown"
            for track in self.root.findall(".//track"):
                if polygon in track:
                    label = track.get("label", "unknown")
                    break
            
            # If not in track, check if it has a direct label attribute
            if label == "unknown":
                label = polygon.get("label", "unknown")
                
            # Parse points string into list of (x, y) tuples
            points_str = polygon.get("points")
            points = []
            if points_str:
                points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
            
            polygon_data = {
                "shape_type": "polygon",
                "label": label,
                "frame": frame,
                "points": points,
                "keyframe": bool(int(polygon.get("keyframe", "0"))),
                "outside": bool(int(polygon.get("outside", "0"))),
                "occluded": bool(int(polygon.get("occluded", "0"))),
                "z_order": int(polygon.get("z_order", "0"))
            }
            polygons_per_frame[frame].append(polygon_data)
            
        return polygons_per_frame
    
    def parse_polylines(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all polyline annotations organized by frame."""
        polylines_per_frame = {}
        
        for polyline in self.root.findall(".//polyline"):
            frame = int(polyline.get("frame"))
            if frame not in polylines_per_frame:
                polylines_per_frame[frame] = []
                
            # Get the track label from parent track element
            # Find the parent track element
            label = "unknown"
            for track in self.root.findall(".//track"):
                if polyline in track:
                    label = track.get("label", "unknown")
                    break
            
            # Parse points string into list of (x, y) tuples
            points_str = polyline.get("points")
            points = []
            if points_str:
                points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
            
            polyline_data = {
                "shape_type": "polyline",
                "label": label,
                "frame": frame,
                "points": points,
                "keyframe": bool(int(polyline.get("keyframe", "0"))),
                "outside": bool(int(polyline.get("outside", "0"))),
                "occluded": bool(int(polyline.get("occluded", "0"))),
                "z_order": int(polyline.get("z_order", "0"))
            }
            polylines_per_frame[frame].append(polyline_data)
            
        return polylines_per_frame
    
    def parse_all_shapes(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all shapes (ellipses, boxes, polygons, polylines) organized by frame."""
        all_shapes_per_frame = {}
        
        # Get all shape types
        ellipses = self.parse_ellipses()
        boxes = self.parse_boxes()
        polygons = self.parse_polygons()
        polylines = self.parse_polylines()
        
        # Combine all shapes by frame
        all_frames = set()
        all_frames.update(ellipses.keys())
        all_frames.update(boxes.keys())
        all_frames.update(polygons.keys())
        all_frames.update(polylines.keys())
        
        for frame in all_frames:
            all_shapes_per_frame[frame] = []
            
            # Add all shapes for this frame
            if frame in ellipses:
                all_shapes_per_frame[frame].extend(ellipses[frame])
            if frame in boxes:
                all_shapes_per_frame[frame].extend(boxes[frame])
            if frame in polygons:
                all_shapes_per_frame[frame].extend(polygons[frame])
            if frame in polylines:
                all_shapes_per_frame[frame].extend(polylines[frame])
                
            # Sort by z_order for proper layering
            all_shapes_per_frame[frame].sort(key=lambda x: x.get("z_order", 0))
        
        return all_shapes_per_frame
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the annotations."""
        all_shapes = self.parse_all_shapes()
        
        total_shapes = sum(len(shapes) for shapes in all_shapes.values())
        shape_type_counts = {}
        label_counts = {}
        frame_count = len(all_shapes)
        
        for frame_shapes in all_shapes.values():
            for shape in frame_shapes:
                shape_type = shape["shape_type"]
                label = shape["label"]
                
                shape_type_counts[shape_type] = shape_type_counts.get(shape_type, 0) + 1
                label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_shapes": total_shapes,
            "total_frames": frame_count,
            "shapes_by_type": shape_type_counts,
            "shapes_by_label": label_counts,
            "unique_labels": list(self.get_labels())
        }
    
    def export_to_json(self, output_file: str):
        """Export all shape data to JSON file."""
        data = {
            "metadata": {
                "source_file": self.xml_file_path,
                "summary": self.get_summary_stats()
            },
            "shapes_by_frame": self.parse_all_shapes()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Shape data exported to {output_file}")


def main():
    """Example usage of the XMLShapeParser."""
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    # Initialize parser
    parser = XMLShapeParser(xml_file)
    
    # Get basic information
    print("=== XML Shape Parser Results ===")
    print(f"Parsing file: {xml_file}")
    
    # Get labels
    labels = parser.get_labels()
    print(f"\nUnique labels found: {labels}")
    
    # Get summary statistics
    stats = parser.get_summary_stats()
    print(f"\nSummary Statistics:")
    print(f"- Total shapes: {stats['total_shapes']}")
    print(f"- Total frames: {stats['total_frames']}")
    print(f"- Shapes by type: {stats['shapes_by_type']}")
    print(f"- Shapes by label: {stats['shapes_by_label']}")
    
    # Parse all shapes
    all_shapes = parser.parse_all_shapes()
    
    # Show sample data for first few frames
    print(f"\nSample data from first 3 frames:")
    for frame_num in sorted(all_shapes.keys())[:3]:
        shapes = all_shapes[frame_num]
        print(f"\nFrame {frame_num}: {len(shapes)} shapes")
        for i, shape in enumerate(shapes[:2]):  # Show first 2 shapes per frame
            print(f"  Shape {i+1}: {shape['shape_type']} - {shape['label']}")
            if shape['shape_type'] == 'ellipse':
                print(f"    Center: ({shape['cx']:.2f}, {shape['cy']:.2f}), Radii: ({shape['rx']:.2f}, {shape['ry']:.2f})")
            elif shape['shape_type'] == 'box':
                print(f"    Box: ({shape['xtl']:.2f}, {shape['ytl']:.2f}) to ({shape['xbr']:.2f}, {shape['ybr']:.2f})")
            elif shape['shape_type'] in ['polygon', 'polyline']:
                print(f"    Points: {len(shape['points'])} points")
    
    # Export to JSON
    parser.export_to_json("parsed_shapes.json")
    
    print(f"\n=== Parsing complete! ===")


if __name__ == "__main__":
    main()