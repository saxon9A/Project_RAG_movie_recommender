def movie_to_text(row) -> str:
    return f"""
    Title: {row['title']}
    Overview: {row['overview']}
    Genres: {genres}
    Rating: {row['vote_average']}
    """