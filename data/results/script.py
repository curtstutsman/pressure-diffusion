import csv
data = []
with open("varying_alpha_facebook.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append((float(row[1]), float(row[2].strip("[]"))))

out = "\n".join(
    f"({ ', '.join(f'{x:.8f}'.rstrip('0').rstrip('.') for x in t) })"
    for t in data
)
print(out)

Where to find large sparse datasets, most sparse ones on SNAPS have around 4x as many edges
Potential random graphs to try:
    -Barabasi-Albert: prefers to attach to other nodes w/ high degree (simulates influencers)
    -Watts-Strogatz: high clustering and short average path lengths
    -