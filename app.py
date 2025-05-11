[之前的代码保持不变，仅修改 filter_dissimilar_songs 函数中的相似度计算部分]

def filter_dissimilar_songs(reference_songs, candidate_songs):
    """Filter tracks that are significantly different from the reference group"""
    if not reference_songs or not candidate_songs:
        return []
    
    # Get reference group feature vectors
    reference_features = []
    reference_language = None
    reference_artist_era = None
    reference_genre_main = None
    reference_genre = None
    reference_era = None
    reference_artists = set()  # 收集参考组的所有歌手
    reference_tags = None
    
    # 首先收集所有参考组的歌手
    for song in reference_songs:
        if song in track_features_map:
            artist_name = track_features_map[song].get('artist', None)
            if artist_name:
                # 将歌手名转为小写并添加到集合中
                reference_artists.add(artist_name.lower())
    
    logger.info(f"参考组(A组)包含的歌手: {', '.join(reference_artists)}")
    
    # 然后收集其他特征
    for song in reference_songs:
        if song in track_features_map:
            features = track_features_map[song]['features']
            feature_vector = [
                features['danceability'], features['energy'], 
                features['speechiness'], features['acousticness'], 
                features['instrumentalness'], features['liveness'], 
                features['valence']
            ]
            reference_features.append(feature_vector)
            
            # 收集参考组的特征
            if reference_language is None:
                reference_language = features.get('language', 'unknown')
            if reference_artist_era is None and 'artist_era' in features and features['artist_era'] is not None:
                reference_artist_era = features['artist_era']
            if reference_genre_main is None:
                reference_genre_main = features.get('genre_main', 'unknown')
            if reference_genre is None:
                reference_genre = features.get('genre', 'unknown')
            if reference_era is None:
                reference_era = features.get('era', 'unknown')
            
            if reference_tags is None:
                reference_tags = set()
                if 'tags' in features and isinstance(features['tags'], list):
                    reference_tags = set([t['name'].lower() for t in features['tags'] if 'name' in t])
    
    if not reference_features:
        return []
    
    # Calculate reference group average vector
    reference_avg = np.mean(reference_features, axis=0)
    
    # Calculate similarity for each candidate song
    dissimilar_songs = []
    for song in candidate_songs:
        if song in track_features_map:
            features = track_features_map[song]['features']
            artist_name = track_features_map[song].get('artist', None)
            
            # 1. 首先检查是否是同一歌手的歌曲
            if artist_name and artist_name.lower() in reference_artists:
                logger.info(f"歌曲 '{track_features_map[song]['name']}' 的歌手 '{artist_name}' 在A组中出现过，默认为相似")
                continue
            
            # 初始化相似度分数（0-1，1表示完全相似）
            similarity_score = 1.0
            reasons = []
            
            # 2. 优先判断语言差异（权重最高）
            song_language = features.get('language', 'unknown')
            if song_language != reference_language and reference_language != 'unknown' and song_language != 'unknown':
                similarity_score -= 0.5  # 语言不同，显著降低相似度
                reasons.append(f"语言不同: {song_language} vs {reference_language}")
            
            # 3. 判断歌手时代差异
            artist_era = features.get('artist_era')
            if artist_era and reference_artist_era:
                era_diff = abs(artist_era - reference_artist_era)
                if era_diff > 20:  # 只有超过20年才考虑时代差异
                    similarity_score -= 0.3  # 时代差异大，显著降低相似度
                    reasons.append(f"歌手时代差距大: {artist_era} vs {reference_artist_era} (相差{era_diff}年)")
            
            # 4. 如果语言和时代都没有显著差异，再考虑其他因素
            if similarity_score > 0.5:
                # 主风格相似度判断（权重较低）
                if features.get('genre_main', 'unknown') != reference_genre_main and reference_genre_main != 'unknown' and features.get('genre_main', 'unknown') != 'unknown':
                    similarity_score -= 0.15  # 主风格不同，轻微降低相似度
                    reasons.append(f"主风格不同: {features.get('genre_main', 'unknown')} vs {reference_genre_main}")
                
                # 音乐特征向量距离判断（权重最低）
                feature_vector = [
                    features['danceability'], features['energy'], 
                    features['speechiness'], features['acousticness'], 
                    features['instrumentalness'], features['liveness'], 
                    features['valence']
                ]
                distance = np.linalg.norm(np.array(feature_vector) - reference_avg)
                if distance > 2.0:  # 提高阈值，降低影响
                    similarity_score -= 0.1
                    reasons.append(f"音乐特征差异大: 距离值 {distance:.2f}")
            
            # 5. 判断是否为显著不同
            if similarity_score < 0.5:  # 保持阈值不变
                combined_reason = " | ".join(reasons)
                dissimilar_songs.append({
                    "id": song, 
                    "reason": combined_reason,
                    "similarity_score": round(similarity_score, 2)
                })
    
    return dissimilar_songs

[其余代码保持不变]