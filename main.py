import heapq

# Dijkstra's Algorithm to calculate the shortest path from a source to all nodes
def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Calculate total travel time for all universities to a specific bar
def total_travel_time(graph, universities, bar):
    total_time = 0
    for uni in universities:
        distances = dijkstra(graph, uni)
        if distances[bar] != float('infinity'):
            total_time = max(total_time, distances[bar])  # Take maximum travel time among universities
        else:
            print(f"{uni} cannot reach {bar}")
    return total_time

# Calculate weighted score for a bar based on time, rating, and price
def calculate_weighted_score(time, rating, price, weight_time, weight_rating, weight_price):
    return (weight_time * time) + (weight_rating * rating) + (weight_price * price)

# Main function to find the top three optimal bars
def find_optimal_bar(graph, universities, bars, weights, bar_ratings, bar_prices):
    # Weights for each factor
    weight_time, weight_rating, weight_price = weights
    
    # Store scores for each bar
    bar_scores = []
    
    for bar in bars:
        # Get individual metrics
        travel_time = total_travel_time(graph, universities, bar)
        rating = bar_ratings.get(bar, 0)
        price = bar_prices.get(bar, 0)
        
        # Calculate the weighted score
        score = calculate_weighted_score(travel_time, rating, price, weight_time, weight_rating, weight_price)
        
        # Store as tuple (score, bar) for sorting
        bar_scores.append((score, bar))
    
    # Sort bars by score and take the top three
    bar_scores.sort()
    return bar_scores[:3]

def main():
    # Example setup for bar ratings and prices
    bar_ratings = {
        'jarra_pipa': 8, 'cappuccino': 7, 'el_tigre': 9, 'fontana_oro': 6, 'baton_rouge': 7, 'radio_rooftop': 8,
        'inclan': 6, 'mambo': 7, 'gato': 8, 'bar_luis': 5, 'bar_v': 7, 'churruca': 9, 'bar_sidi': 6, 'madriz_bar': 8,
        'labrador': 7, 's10bar': 8, 'bar_armando': 7, 'terraza': 6, 'casa_malicia': 9, 'cuevita': 8, 'bar_cruz': 6,
        'moreno': 7, 'serrano80': 8, 'jurucha': 6, 'el_41': 7
    }

    bar_prices = {
        'jarra_pipa': 15, 'cappuccino': 12, 'el_tigre': 10, 'fontana_oro': 9, 'baton_rouge': 11, 'radio_rooftop': 14,
        'inclan': 10, 'mambo': 11, 'gato': 13, 'bar_luis': 8, 'bar_v': 10, 'churruca': 13, 'bar_sidi': 10, 'madriz_bar': 12,
        'labrador': 11, 's10bar': 12, 'bar_armando': 9, 'terraza': 10, 'casa_malicia': 14, 'cuevita': 12, 'bar_cruz': 9,
        'moreno': 11, 'serrano80': 13, 'jurucha': 12, 'el_41': 10
    }

    # Example graph setup
    graph = {
        # Universities connected to metro stations
        'ietower': {'begona': 8},
        'iemm': {'av_america': 8},
        'reyjuan': {'man_becerra': 2},
        'uc3m': {'pta_toledo': 1},
        'complu': {'ciud_uni': 1},
        
        # Metro stations connected to each other (within the same metro line)
        'begona': {'ietower': 8, 'plaza_castilla': 4},
        'plaza_castilla': {'begona': 8, 'av_america': 9, 'nuev_min': 4},
        'av_america': {'iemm': 8, 'plaza_castilla': 9, 'man_becerra': 3, 'goya': 5, 'nuev_min': 4, 'greg_mar': 3, 'nun_balb': 2},
        'man_becerra': {'reyjuan': 2, 'av_america': 3, 'goya': 1},
        'goya': {'man_becerra': 1, 'av_america': 5, 'serrano': 2, 'pri_vergara': 1, 'jarra_pipa': 1},
        'pri_vergara': {'goya': 1, 'jarra_pipa': 7, 'nun_balb': 2, 'retiro': 1},
        'retiro': {'pri_vergara': 1, 'cappuccino': 2, 'banco_esp': 2},
        'banco_esp': {'el_tigre': 8, 'retiro': 2, 'sol': 2},
        'sol': {'banco_esp': 2, 'gran_via': 1, 'tirso': 2, 'opera': 1, 'callao': 1, 'fontana_oro': 1, 'baton_rouge': 4, 'radio_rooftop': 6, 'inclan': 5, 'mambo': 5},
        'opera': {'sol': 1, 'latina': 2, 'callao': 2, 'san_bernardo': 4},
        'san_bernardo': {'opera': 4, 'arguelles': 3, 'bilbao': 1, 'quevedo': 1, 'gato': 6},
        'quevedo': {'san_bernardo': 1, 'canal': 2, 'bar_luis': 6},
        'canal': {'quevedo': 2, 'cuatro_cam': 2, 'islas_fil': 2, 'greg_mar': 2, 'bar_luis': 8},
        'cuatro_cam': {'canal': 2, 'guzman': 2, 'bilbao': 5, 'nuev_min': 2},
        'nuev_min': {'plaza_castilla': 4, 'cuatro_cam': 2, 'greg_mar': 2, 'av_america': 4},
        'greg_mar': {'nuev_min': 2, 'av_america': 3, 'canal': 2, 'alonso_mar': 2},
        'alonso_mar': {'greg_mar': 2, 'nun_balb': 3, 'serrano': 2, 'chueca': 2, 'tribunal': 1, 'bilbao': 2, 'bar_v': 10},
        'tribunal': {'alonso_mar': 1, 'bilbao': 1, 'principe_pio': 3, 'gran_via': 2, 'churruca': 6, 'gato': 6, 'bar_sidi': 7}, 
        'principe_pio': {'tribunal': 3, 'arguelles': 4},
        'arguelles': {'principe_pio': 4, 'callao': 3, 'san_bernardo': 3, 'moncloa': 1, 'madriz_bar': 5},
        'moncloa': {'arguelles': 1, 'ciud_uni': 3, 'labrador': 9, 's10bar': 11, 'bar_armando': 9},
        'ciud_uni': {'moncloa': 3, 'guzman': 3, 'complu': 1},
        'guzman': {'ciud_uni': 3, 'cuatro_cam': 2, 'islas_fil': 3},
        'islas_fil': {'guzman': 3, 'canal': 2, 's10bar': 6, 'bar_armando': 6, 'madriz_bar': 11, 'labrador': 5},
        'bilbao': {'cuatro_cam': 5, 'alonso_mar': 2, 'tribunal': 1, 'san_bernardo': 1, 'churruca': 4},
        'gran_via': {'tribunal': 2, 'chueca': 2, 'sol': 1, 'callao': 1},
        'callao': {'gran_via': 1, 'sol': 1, 'opera': 2, 'arguelles': 3},
        'tirso': {'sol': 2, 'terraza': 6, 'casa_malicia': 9, 'cuevita': 8, 'bar_cruz': 5, 'moreno': 10},
        'pta_toledo': {'uc3m': 1, 'latina': 1},
        'latina': {'pta_toledo': 1, 'opera': 2, 'terraza': 7, 'casa_malicia': 10, 'cuevita': 7, 'bar_cruz': 1, 'moreno': 8},
        'chueca': {'gran_via': 2, 'alonso_mar': 2, 'bar_v': 1, 'el_tigre': 5, 'bar_sidi': 7},
        'nun_balb': {'chueca': 3, 'av_america': 2, 'pri_vergara': 2, 'serrano80': 8, 'jurucha': 11},
        'serrano': {'goya': 2, 'alonso_mar': 2, 'el_41': 2},



        # Bars connected to metro stations by walking times
        'jarra_pipa': {'goya': 1, 'pri_vergara': 7},
        'cappuccino': {'retiro': 2},
        'el_tigre': {'banco_esp': 8, 'chueca': 5},
        'fontana_oro': {'sol':1},
        'baton_rouge': {'sol': 4},
        'radio_rooftop': {'sol': 6},
        'inclan': {'sol': 5},
        'mambo': {'sol': 5},
        'gato': {'san_bernardo': 6, 'tribunal': 6},
        'bar_luis': {'quevedo': 6, 'canal': 6},
        'bar_v': {'alonso_mar': 10, 'chueca': 1},
        'churruca': {'tribunal': 6, 'bilbao': 4},
        'bar_sidi': {'tribunal': 7, 'chueca': 7},
        'madriz_bar': {'arguelles': 5, 'islas_fil': 11},
        'labrador': {'moncloa': 9, 'islas_fil': 5},
        's10bar': {'moncloa': 11, 'islas_fil': 6},
        'bar_armando': {'moncloa': 9, 'islas_fil': 6},
        'terraza': {'tirso': 6, 'latina': 7},
        'casa_malicia': {'tirso': 9, 'latina': 10},
        'cuevita': {'tirso': 8, 'latina': 7},
        'bar_cruz': {'tirso': 5, 'latina': 1},
        'moreno': {'tirso': 10, 'latina': 8},
        'serrano80': {'nun_balb': 8, 'serrano': 11},
        'jurucha': {'nun_balb': 11, 'serrano': 6},
        'el_41': {'serrano': 2},
    }

    # User input for weights
    weight_time = float(input("Enter weight for time (0-1): "))
    weight_rating = float(input("Enter weight for rating (0-1): "))
    weight_price = float(input("Enter weight for price (0-1): "))

    # Normalizing weights to add up to 1
    total_weight = weight_time - weight_rating + weight_price
    weights = (weight_time / total_weight, weight_rating / total_weight, weight_price / total_weight)

    # List of all universities and bars
    universities = ['ietower', 'iemm', 'reyjuan', 'uc3m', 'complu']
    bars = [
        'jarra_pipa', 'cappuccino', 'el_tigre', 'fontana_oro', 'baton_rouge', 'radio_rooftop', 'inclan', 'mambo', 'gato',
        'bar_luis', 'bar_v', 'churruca', 'bar_sidi', 'madriz_bar', 'labrador', 's10bar', 'bar_armando', 'terraza',
        'casa_malicia', 'cuevita', 'bar_cruz', 'moreno', 'serrano80', 'jurucha', 'el_41'
    ]

    """Find the optimal bars
    top_bars = find_optimal_bars(graph, universities, bars, weight_time, weight_rating, weight_price, bar_ratings, bar_prices)
    for i, (score, bar) in enumerate(top_bars, start=1):
        print(f"Top {i} bar: {bar} with a score of {score:.2f}")
        """
    top_bars = find_optimal_bar(graph, universities, bars, weights, bar_ratings, bar_prices)
    for i, (score, bar) in enumerate(top_bars, start=1):
        print(f"Top {i} bar: {bar} with a weighted score of {score:.2f}")

if __name__ == "__main__":
    main()
