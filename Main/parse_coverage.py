import xml.etree.ElementTree as ET
import csv

def parse_coverage(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespace = ''
    if '}' in root.tag:
        namespace = root.tag.split('}')[0] + '}'

    methods_info = []


    for clazz in root.findall(f".//{namespace}class"):
        class_name = clazz.get('name')
        for method in clazz.find(f"{namespace}methods").findall(f"{namespace}method"):
            method_name = method.get('name')
            signature = method.get('signature')
            line_rate = method.get('line-rate')
            branch_rate = method.get('branch-rate')

            lines = method.find(f"{namespace}lines").findall(f"{namespace}line")
            line_numbers = []
            total_hits = 0
            for line in lines:
                number = line.get('number')
                hits = int(line.get('hits'))
                total_hits += hits
                line_numbers.append(number)

            method_info = {
                'class_name': class_name,
                'method_name': method_name,
                'signature': signature,
                'line_rate': line_rate,
                'branch_rate': branch_rate,
                'total_hits': total_hits,
                'line_numbers': ';'.join(line_numbers)
            }

            methods_info.append(method_info)

    return methods_info

def save_to_csv(methods_info, csv_file):
    fieldnames = ['class_name', 'method_name', 'signature', 'line_rate', 'branch_rate', 'total_hits', 'line_numbers']

    with open(csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for method in methods_info:
            writer.writerow(method)


if __name__ == "__main__":
    xml_file = 'coverage.xml'
    csv_file = 'methods_coverage.csv'
    methods = parse_coverage(xml_file)
    save_to_csv(methods, csv_file)
