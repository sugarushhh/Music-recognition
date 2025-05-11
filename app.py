from flask import Flask, render_template, request, jsonify, session
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import Counter
import time
import html
import logging
import re
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Last.fm API configuration
API_KEY = 'ac92a37f42f55eee4a5fcd7321e752fe'
API_SECRET = '051d35a38d4f1bfada0be98cf806ce60'
API_BASE_URL = 'http://ws.audioscrobbler.com/2.0/'

# Request retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Global variables to store user data
track_features_map = {}
tag_A = {}
tag_B = {}
artist_info_cache = {}  # 缓存歌手信息

# 增加 genre 主类归一化映射
GENRE_MAIN_CLASS = {
    'pop': 'pop',
    'dance': 'pop',
    'synthpop': 'pop',
    'electropop': 'pop',
    'k-pop': 'pop',
    'j-pop': 'pop',
    'rock': 'rock',
    'classic rock': 'rock',
    'hard rock': 'rock',
    'metal': 'rock',
    'punk': 'rock',
    'alternative rock': 'rock',
    'indie rock': 'rock',
    'jazz': 'jazz',
    'blues': 'jazz',
    'hip hop': 'hiphop',
    'hiphop': 'hiphop',
    'rap': 'hiphop',
    'trap': 'hiphop',
    'classical': 'classical',
    'orchestra': 'classical',
    'folk': 'folk',
    'country': 'folk',
    'electronic': 'electronic',
    'edm': 'electronic',
    'techno': 'electronic',
    'house': 'electronic',
    'trance': 'electronic',
    'dubstep': 'electronic',
    'r&b': 'rnb',
    'rnb': 'rnb',
    'soul': 'rnb',
    'funk': 'rnb',
    'reggae': 'reggae',
    'latin': 'latin',
    'salsa': 'latin',
    'flamenco': 'latin',
    'indie': 'indie',
    'alternative': 'alternative',
}

# 添加时代映射
ERA_MAPPING = {
    '50s': '1950s',
    '60s': '1960s',
    '70s': '1970s',
    '80s': '1980s',
    '90s': '1990s',
    '2000s': '2000s',
    '2010s': '2010s',
    '2020s': '2020s',
}

# 歌手年代映射 - 用于估计歌手的活跃年代
ARTIST_ERA = {
    # 1950s
    'elvis presley': 1950,
    'chuck berry': 1950,
    'buddy holly': 1950,
    'little richard': 1950,
    'johnny cash': 1950,
    
    # 1960s
    'the beatles': 1960,
    'the rolling stones': 1960,
    'bob dylan': 1960,
    'the beach boys': 1960,
    'jimi hendrix': 1960,
    'the doors': 1960,
    
    # 1970s
    'led zeppelin': 1970,
    'pink floyd': 1970,
    'queen': 1970,
    'david bowie': 1970,
    'elton john': 1970,
    'fleetwood mac': 1970,
    'abba': 1970,
    
    # 1980s
    'michael jackson': 1980,
    'madonna': 1980,
    'prince': 1980,
    'u2': 1980,
    'guns n roses': 1980,
    'bon jovi': 1980,
    'whitney houston': 1980,
    
    # 1990s
    'nirvana': 1990,
    'radiohead': 1990,
    'tupac shakur': 1990,
    'the notorious b.i.g.': 1990,
    'spice girls': 1990,
    'backstreet boys': 1990,
    'mariah carey': 1990,
    
    # 2000s
    'eminem': 2000,
    'beyonce': 2000,
    'coldplay': 2000,
    'rihanna': 2000,
    'kanye west': 2000,
    'lady gaga': 2000,
    'justin timberlake': 2000,
    
    # 2010s
    'taylor swift': 2010,
    'ed sheeran': 2010,
    'drake': 2010,
    'adele': 2010,
    'bruno mars': 2010,
    'ariana grande': 2010,
    'billie eilish': 2010,
    
    # 2020s
    'doja cat': 2020,
    'olivia rodrigo': 2020,
    'the weeknd': 2020,
    'bad bunny': 2020,
    'bts': 2020,
    'dua lipa': 2020,
    'harry styles': 2020,
}

def safe_api_call(method, params, retries=MAX_RETRIES):
    """Safely call the Last.fm API with error handling and retries"""
    params['api_key'] = API_KEY
    params['format'] = 'json'
    params['method'] = method
    
    for attempt in range(retries):
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                logger.warning(f"API call failed, retrying ({attempt+1}/{retries}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"API call failed after multiple retries: {e}")
                raise
        except Exception as e:
            logger.error(f"Other API error: {e}")
            raise

def is_api_available():
    """Check if Last.fm API is available"""
    try:
        # Try a simple API call to check connection
        result = safe_api_call('chart.getTopArtists', {'limit': 1})
        return True
    except Exception as e:
        logger.error(f"API unavailable: {e}")
        return False

def get_track_info(track_name, artist_name=None):
    """Get track information from Last.fm API"""
    params = {'track': track_name}
    if artist_name:
        params['artist'] = artist_name
        
    try:
        result = safe_api_call('track.getInfo', params)
        return result
    except Exception as e:
        logger.error(f"Error getting track info: {e}")
        return None

def get_artist_info(artist_name):
    """获取歌手信息，包括标签和简介"""
    # 检查缓存
    if artist_name.lower() in artist_info_cache:
        return artist_info_cache[artist_name.lower()]
    
    try:
        result = safe_api_call('artist.getInfo', {'artist': artist_name})
        
        if 'artist' in result:
            # 缓存结果
            artist_info_cache[artist_name.lower()] = result['artist']
            return result['artist']
        return None
    except Exception as e:
        logger.error(f"Error getting artist info: {e}")
        return None

def estimate_artist_era(artist_name):
    """估计歌手的活跃年代"""
    # 检查预定义的歌手年代
    artist_lower = artist_name.lower()
    if artist_lower in ARTIST_ERA:
        return ARTIST_ERA[artist_lower]
    
    # 从Last.fm获取歌手信息
    artist_info = get_artist_info(artist_name)
    if not artist_info:
        return None
    
    # 从标签中提取年代信息
    if 'tags' in artist_info and 'tag' in artist_info['tags']:
        tags = artist_info['tags']['tag']
        if isinstance(tags, dict):
            tags = [tags]
        
        for tag in tags:
            tag_name = tag.get('name', '').lower()
            # 检查标签中是否包含年代信息
            decade_match = re.search(r'(19\d0s|20\d0s)', tag_name)
            if decade_match:
                decade = decade_match.group(1)
                return int(decade[:4])
            
            # 检查特定年代标签
            if '50s' in tag_name or '1950s' in tag_name:
                return 1950
            elif '60s' in tag_name or '1960s' in tag_name:
                return 1960
            elif '70s' in tag_name or '1970s' in tag_name:
                return 1970
            elif '80s' in tag_name or '1980s' in tag_name:
                return 1980
            elif '90s' in tag_name or '1990s' in tag_name:
                return 1990
            elif '00s' in tag_name or '2000s' in tag_name:
                return 2000
            elif '10s' in tag_name or '2010s' in tag_name:
                return 2010
            elif '20s' in tag_name or '2020s' in tag_name:
                return 2020
    
    # 从简介中提取年代信息
    if 'bio' in artist_info and 'content' in artist_info['bio']:
        bio = artist_info['bio']['content']
        # 查找形如"formed in 1985"或"started in 1990s"的文本
        formed_match = re.search(r'formed in (\d{4})', bio, re.IGNORECASE)
        if formed_match:
            year = int(formed_match.group(1))
            return (year // 10) * 10  # 返回对应的十年
        
        started_match = re.search(r'started in (\d{4})', bio, re.IGNORECASE)
        if started_match:
            year = int(started_match.group(1))
            return (year // 10) * 10
        
        # 查找任何四位数年份，取最早的一个作为参考
        years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', bio)
        if years:
            earliest_year = min(int(y) for y in years)
            if 1950 <= earliest_year <= datetime.now().year:
                return (earliest_year // 10) * 10
    
    # 如果无法确定，返回None
    return None

def search_track(track_name):
    """Search for tracks on Last.fm with improved relevance"""
    try:
        # 检查是否是在搜索歌手
        artist_result = safe_api_call('artist.search', {'artist': track_name, 'limit': 1})
        if ('results' in artist_result and 'artistmatches' in artist_result['results'] and 
            'artist' in artist_result['results']['artistmatches']):
            artist_matches = artist_result['results']['artistmatches']['artist']
            if isinstance(artist_matches, dict):
                artist_matches = [artist_matches]
            
            # 如果找到了精确匹配的歌手，搜索该歌手的热门歌曲
            for artist in artist_matches:
                if artist.get('name', '').lower() == track_name.lower():
                    logger.info(f"找到精确匹配的歌手: {artist.get('name')}")
                    # 获取该歌手的热门歌曲
                    top_tracks = safe_api_call('artist.getTopTracks', {'artist': artist.get('name'), 'limit': 5})
                    if 'toptracks' in top_tracks and 'track' in top_tracks['toptracks']:
                        tracks = top_tracks['toptracks']['track']
                        if isinstance(tracks, dict):
                            tracks = [tracks]
                        
                        # 转换为标准格式
                        results = []
                        for track in tracks:
                            results.append({
                                'name': track.get('name', ''),
                                'artist': artist.get('name', ''),
                                'listeners': track.get('listeners', '0')
                            })
                        return results
        
        # 常规歌曲搜索
        result = safe_api_call('track.search', {'track': track_name, 'limit': 10})
        
        if 'results' in result and 'trackmatches' in result['results'] and 'track' in result['results']['trackmatches']:
            tracks = result['results']['trackmatches']['track']
            # Last.fm might return a single dict for one result, convert to list
            if isinstance(tracks, dict):
                tracks = [tracks]
            
            # 对结果进行排序：完全匹配 > 开头匹配 > 包含匹配
            exact_matches = []
            starts_with_matches = []
            contains_matches = []
            
            for track in tracks:
                track_name_lower = track.get('name', '').lower()
                search_term_lower = track_name.lower()
                
                if track_name_lower == search_term_lower:
                    exact_matches.append(track)
                elif track_name_lower.startswith(search_term_lower):
                    starts_with_matches.append(track)
                else:
                    contains_matches.append(track)
            
            # 按优先级合并结果
            sorted_tracks = exact_matches + starts_with_matches + contains_matches
            return sorted_tracks[:10]  # 返回前10个结果，后面会进一步筛选
        return []
    except Exception as e:
        logger.error(f"Error searching track: {e}")
        return []

def get_track_tags(track_name, artist_name):
    """Get tags for a track from Last.fm API"""
    try:
        result = safe_api_call('track.getTopTags', {'track': track_name, 'artist': artist_name})
        
        if 'toptags' in result and 'tag' in result['toptags']:
            tags = result['toptags']['tag']
            # Last.fm might return a single dict for one tag, convert to list
            if isinstance(tags, dict):
                tags = [tags]
            return tags
        return []
    except Exception as e:
        logger.error(f"Error getting track tags: {e}")
        return []

def get_similar_tracks(track_name, artist_name):
    """Get similar tracks from Last.fm API"""
    try:
        result = safe_api_call('track.getSimilar', {'track': track_name, 'artist': artist_name, 'limit': 10})
        
        if 'similartracks' in result and 'track' in result['similartracks']:
            similar_tracks = result['similartracks']['track']
            # Last.fm might return a single dict for one result, convert to list
            if isinstance(similar_tracks, dict):
                similar_tracks = [similar_tracks]
            return similar_tracks
        return []
    except Exception as e:
        logger.error(f"Error getting similar tracks: {e}")
        return []

def extract_track_features(track_info):
    """Extract and normalize musical features from track information and tags"""
    if not track_info:
        return None
    
    # Get track tags
    artist_name = track_info.get('artist', {}).get('name', '')
    track_name = track_info.get('name', '')
    
    tags = get_track_tags(track_name, artist_name)
    
    # 调试信息：打印获取到的标签
    logger.info(f"处理歌曲: {track_name} - {artist_name}")
    logger.info(f"获取到的标签数量: {len(tags)}")
    for tag in tags[:10]:  # 打印前10个标签
        logger.info(f"  - {tag.get('name', '')}")
    
    # 获取歌手年代
    artist_era = estimate_artist_era(artist_name)
    logger.info(f"歌手 {artist_name} 的估计年代: {artist_era if artist_era else '未知'}")
    
    # Initialize features with default values
    features = {
        'danceability': 0.5,
        'energy': 0.5,
        'acousticness': 0.5,
        'instrumentalness': 0.5,
        'liveness': 0.5,
        'valence': 0.5,
        'speechiness': 0.5,
        'tempo': 120,  # Default tempo
        'genre': 'unknown',
        'language': 'unknown',
        'era': 'unknown',  # 添加时代特征
        'artist_era': artist_era,  # 添加歌手年代
    }
    
    # Map common tags to musical features, genre, and language
    tag_mapping = {
        # 舞曲特性
        'dance': ('danceability', 0.8),
        'danceable': ('danceability', 0.8),
        'edm': ('danceability', 0.9),
        'club': ('danceability', 0.8),
        'party': ('danceability', 0.7),
        
        # 能量特性
        'energetic': ('energy', 0.9),
        'high energy': ('energy', 0.9),
        'powerful': ('energy', 0.8),
        'intense': ('energy', 0.9),
        'calm': ('energy', 0.2),
        'chill': ('energy', 0.3),
        'relaxing': ('energy', 0.2),
        'mellow': ('energy', 0.3),
        'soft': ('energy', 0.2),
        
        # 原声特性
        'acoustic': ('acousticness', 0.9),
        'unplugged': ('acousticness', 0.8),
        'live acoustic': ('acousticness', 0.9),
        
        # 器乐特性
        'instrumental': ('instrumentalness', 0.9),
        'no vocals': ('instrumentalness', 0.9),
        'orchestra': ('instrumentalness', 0.8),
        
        # 现场特性
        'live': ('liveness', 0.9),
        'concert': ('liveness', 0.8),
        'recorded live': ('liveness', 0.9),
        
        # 情感特性
        'happy': ('valence', 0.9),
        'sad': ('valence', 0.2),
        'melancholy': ('valence', 0.3),
        'uplifting': ('valence', 0.8),
        'dark': ('valence', 0.2),
        'emotional': ('valence', 0.4),
        'upbeat': ('valence', 0.8),
        
        # 语音特性
        'rap': ('speechiness', 0.9),
        'spoken word': ('speechiness', 0.9),
        'talking': ('speechiness', 0.8),
        'vocals': ('speechiness', 0.7),
        
        # 速度特性
        'fast': ('tempo', 160),
        'slow': ('tempo', 80),
        'medium tempo': ('tempo', 120),
        
        # 流派
        'pop': ('genre', 'pop'),
        'rock': ('genre', 'rock'),
        'jazz': ('genre', 'jazz'),
        'blues': ('genre', 'blues'),
        'hip hop': ('genre', 'hiphop'),
        'hip-hop': ('genre', 'hiphop'),
        'rap': ('genre', 'hiphop'),
        'classical': ('genre', 'classical'),
        'metal': ('genre', 'metal'),
        'folk': ('genre', 'folk'),
        'country': ('genre', 'country'),
        'electronic': ('genre', 'electronic'),
        'edm': ('genre', 'electronic'),
        'techno': ('genre', 'electronic'),
        'house': ('genre', 'electronic'),
        'trance': ('genre', 'electronic'),
        'dubstep': ('genre', 'electronic'),
        'r&b': ('genre', 'rnb'),
        'rnb': ('genre', 'rnb'),
        'soul': ('genre', 'rnb'),
        'funk': ('genre', 'rnb'),
        'reggae': ('genre', 'reggae'),
        'latin': ('genre', 'latin'),
        'salsa': ('genre', 'latin'),
        'flamenco': ('genre', 'latin'),
        'indie': ('genre', 'indie'),
        'alternative': ('genre', 'alternative'),
        'punk': ('genre', 'punk'),
        'grunge': ('genre', 'grunge'),
        'disco': ('genre', 'disco'),
        'k-pop': ('genre', 'k-pop'),
        'j-pop': ('genre', 'j-pop'),
        'c-pop': ('genre', 'c-pop'),
        'mandopop': ('genre', 'c-pop'),
        'cantopop': ('genre', 'c-pop'),
        
        # 语言
        'english': ('language', 'english'),
        'chinese': ('language', 'chinese'),
        'mandarin': ('language', 'chinese'),
        'cantonese': ('language', 'chinese'),
        'japanese': ('language', 'japanese'),
        'french': ('language', 'french'),
        'german': ('language', 'german'),
        'spanish': ('language', 'spanish'),
        'korean': ('language', 'korean'),
        'italian': ('language', 'italian'),
        'russian': ('language', 'russian'),
        'portuguese': ('language', 'portuguese'),
        'hindi': ('language', 'hindi'),
        'arabic': ('language', 'arabic'),
        'turkish': ('language', 'turkish'),
        'thai': ('language', 'thai'),
        'vietnamese': ('language', 'vietnamese'),
        
        # 时代
        '50s': ('era', '1950s'),
        '60s': ('era', '1960s'),
        '70s': ('era', '1970s'),
        '80s': ('era', '1980s'),
        '90s': ('era', '1990s'),
        '2000s': ('era', '2000s'),
        '2010s': ('era', '2010s'),
        '2020s': ('era', '2020s'),
        'oldies': ('era', 'oldies'),
        'classic': ('era', 'classic'),
        'retro': ('era', 'retro'),
        'vintage': ('era', 'vintage'),
        'modern': ('era', 'modern'),
        'contemporary': ('era', 'contemporary'),
    }
    
    # Extract tag names and count
    tag_count = 0
    if tags:
        for tag in tags:
            tag_name = tag.get('name', '').lower()
            tag_count += 1
            # Map tag to feature/genre/language
            for keyword, (feature, value) in tag_mapping.items():
                if keyword in tag_name:
                    features[feature] = value
    
    # Include track listeners and playcount as popularity metrics
    features['popularity'] = float(track_info.get('listeners', 0)) / 1000000  # Normalize by million
    
    # If duration is available, convert to seconds
    if 'duration' in track_info:
        try:
            features['duration'] = int(track_info['duration']) / 1000  # Convert ms to seconds
        except (ValueError, TypeError):
            features['duration'] = 0
    
    # 归一化 genre 到主类
    genre = features.get('genre', 'unknown')
    features['genre_main'] = GENRE_MAIN_CLASS.get(genre, genre)
    
    # 保存原始标签列表用于重叠度判定
    features['tags'] = tags
    
    # 调试信息：打印提取的特征
    logger.info("提取的特征:")
    for key, value in features.items():
        if key != 'tags':  # 不打印完整标签列表
            logger.info(f"  - {key}: {value}")
    
    return features

def analyze_music_features(features_list):
    """Analyze a group of tracks' features and generate tags"""
    if not features_list:
        return {}
    
    # Calculate averages
    avg_features = {}
    for key in ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'speechiness', 'tempo', 'popularity']:
        values = [features[key] for features in features_list if key in features]
        if values:
            avg_features[key] = sum(values) / len(values)
    
    # Generate descriptive tags
    tags = {}
    
    if avg_features['energy'] > 0.7:
        tags['energy'] = 'high'
    elif avg_features['energy'] < 0.4:
        tags['energy'] = 'low'
    else:
        tags['energy'] = 'medium'
        
    if avg_features['danceability'] > 0.7:
        tags['danceability'] = 'high'
    elif avg_features['danceability'] < 0.4:
        tags['danceability'] = 'low'
    else:
        tags['danceability'] = 'medium'
    
    if avg_features['tempo'] > 120:
        tags['tempo'] = 'fast'
    elif avg_features['tempo'] < 90:
        tags['tempo'] = 'slow'
    else:
        tags['tempo'] = 'medium'
    
    if avg_features['acousticness'] > 0.7:
        tags['style'] = 'acoustic'
    elif avg_features['instrumentalness'] > 0.5:
        tags['style'] = 'instrumental'
    elif avg_features['speechiness'] > 0.4:
        tags['style'] = 'speech-heavy'
    
    if avg_features['valence'] > 0.7:
        tags['mood'] = 'positive'
    elif avg_features['valence'] < 0.3:
        tags['mood'] = 'negative'
    else:
        tags['mood'] = 'neutral'
    
    if avg_features['popularity'] > 0.5:
        tags['popularity'] = 'mainstream'
    elif avg_features['popularity'] < 0.1:
        tags['popularity'] = 'niche'
    else:
        tags['popularity'] = 'moderate'
    
    # 添加主要流派、语言和歌手年代
    genre_counter = Counter()
    language_counter = Counter()
    era_counter = Counter()
    artist_era_counter = Counter()
    
    for features in features_list:
        if 'genre_main' in features and features['genre_main'] != 'unknown':
            genre_counter[features['genre_main']] += 1
        if 'language' in features and features['language'] != 'unknown':
            language_counter[features['language']] += 1
        if 'era' in features and features['era'] != 'unknown':
            era_counter[features['era']] += 1
        if 'artist_era' in features and features['artist_era'] is not None:
            artist_era_counter[features['artist_era']] += 1
    
    # 添加最常见的流派、语言和时代
    if genre_counter:
        tags['genre_main'] = genre_counter.most_common(1)[0][0]
    if language_counter:
        tags['language'] = language_counter.most_common(1)[0][0]
    if era_counter:
        tags['era'] = era_counter.most_common(1)[0][0]
    if artist_era_counter:
        tags['artist_era'] = artist_era_counter.most_common(1)[0][0]
    
    # Add average features
    tags['avg_features'] = avg_features
    
    return tags

def calculate_era_similarity_score(artist_era_a, artist_era_b):
    """计算歌手年代的相似度分数，返回0-1之间的值，1表示完全相似"""
    if artist_era_a is None or artist_era_b is None:
        return 0.5  # 如果任一年代未知，返回中等相似度
    
    # 计算年代差距
    era_diff = abs(artist_era_a - artist_era_b)
    
    # 差距在30年以内认为是相似的，差距越小相似度越高
    if era_diff <= 30:
        # 将30年内的差距映射到0.5-1.0的相似度范围
        # 0年差距 -> 1.0相似度
        # 30年差距 -> 0.5相似度
        return 1.0 - (era_diff / 60.0)
    else:
        # 将大于30年的差距映射到0.0-0.5的相似度范围
        # 30年差距 -> 0.5相似度
        # 90年或更大差距 -> 0.0相似度
        return max(0.0, 0.5 - ((era_diff - 30) / 120.0))

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

def compare_tags(tag_a, tag_b):
    """Compare differences between two tag sets"""
    comparison = {}
    
    # Compare basic tags
    for key in ['energy', 'danceability', 'tempo', 'style', 'mood', 'popularity', 'genre_main', 'language', 'era', 'artist_era']:
        if key in tag_a and key in tag_b:
            comparison[key] = {
                'tag_a': tag_a[key],
                'tag_b': tag_b[key],
                'different': tag_a[key] != tag_b[key]
            }
    
    # Compare average feature values
    if 'avg_features' in tag_a and 'avg_features' in tag_b:
        feature_diffs = {}
        for feature, value_a in tag_a['avg_features'].items():
            if feature in tag_b['avg_features']:
                diff = abs(value_a - tag_b['avg_features'][feature])
                feature_diffs[feature] = {
                    'tag_a': round(value_a, 2),
                    'tag_b': round(tag_b['avg_features'][feature], 2),
                    'diff': round(diff, 2),
                    'significant': diff > 0.2  # Consider difference significant if > 0.2
                }
        comparison['feature_diffs'] = feature_diffs
    
    return comparison

@app.route('/')
def index():
    """Serve the single page application"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """Return API availability status"""
    status = is_api_available()
    return jsonify({'status': 'available' if status else 'unavailable'})

@app.route('/api/search_track', methods=['POST'])
def api_search_track():
    """Search for tracks and return results with improved relevance"""
    query = request.json.get('query', '')
    if not query:
        return jsonify({'success': False, 'error': 'No search query provided'})
    
    try:
        tracks = search_track(query)
        results = []
        
        # 限制返回5个最相关的结果
        for track in tracks[:5]:
            track_name = track.get('name', '')
            artist_name = track.get('artist', '')
            results.append({
                'name': track_name,
                'artist': artist_name,
                'id': f"{artist_name}:{track_name}"  # Create a unique ID
            })
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"搜索歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_track', methods=['POST'])
def api_process_track():
    """Process a track and extract its features"""
    track_name = request.json.get('track', '')
    artist_name = request.json.get('artist', '')
    song_key = request.json.get('id', '')
    
    if not track_name or not artist_name:
        return jsonify({'success': False, 'error': 'Track and artist required'})
    
    try:
        # Get detailed track info
        track_info = get_track_info(track_name, artist_name)
        
        if not track_info or 'track' not in track_info:
            return jsonify({'success': False, 'error': 'Track not found'})
        
        # Extract features
        features = extract_track_features(track_info['track'])
        
        if not features:
            return jsonify({'success': False, 'error': 'Failed to extract features'})
        
        # Store in global map
        track_features_map[song_key] = {
            'name': track_name,
            'artist': artist_name,
            'features': features
        }
        
        return jsonify({
            'success': True, 
            'name': track_name,
            'artist': artist_name,
            'features': features
        })
    except Exception as e:
        logger.error(f"处理歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze_group', methods=['POST'])
def api_analyze_group():
    """Analyze a group of tracks and generate tags"""
    group_id = request.json.get('group_id', '')
    song_keys = request.json.get('songs', [])
    
    if not song_keys:
        return jsonify({'success': False, 'error': 'No songs provided'})
    
    try:
        # Get features for valid songs
        valid_features = [track_features_map[key]['features'] for key in song_keys if key in track_features_map]
        
        if not valid_features:
            return jsonify({'success': False, 'error': 'No valid songs found'})
        
        # Generate tags
        tags = analyze_music_features(valid_features)
        
        # Store tags globally
        if group_id == 'A':
            global tag_A
            tag_A = tags
        elif group_id == 'B':
            global tag_B
            tag_B = tags
        
        return jsonify({'success': True, 'tags': tags})
    except Exception as e:
        logger.error(f"分析歌曲组时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/filter_songs', methods=['POST'])
def api_filter_songs():
    """Filter dissimilar songs between two groups"""
    group_a_songs = request.json.get('group_a', [])
    group_b_songs = request.json.get('group_b', [])
    
    if not group_a_songs or not group_b_songs:
        return jsonify({'success': False, 'error': 'Both song groups required'})
    
    try:
        # Filter dissimilar songs
        filtered_songs = filter_dissimilar_songs(group_a_songs, group_b_songs)
        
        # Get detailed info for filtered songs
        filtered_details = []
        for song_item in filtered_songs:
            song_key = song_item["id"]
            if song_key in track_features_map:
                filtered_details.append({
                    'id': song_key,
                    'name': track_features_map[song_key]['name'],
                    'artist': track_features_map[song_key]['artist'],
                    'reason': song_item["reason"],
                    'similarity_score': song_item.get("similarity_score", 0)
                })
        
        return jsonify({
            'success': True, 
            'filtered_songs': filtered_details,
            'total_count': len(group_b_songs),
            'filtered_count': len(filtered_songs)
        })
    except Exception as e:
        logger.error(f"筛选歌曲时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/compare_groups', methods=['POST'])
def api_compare_groups():
    """Compare two groups of songs and their tags"""
    try:
        # Compare tag_A and tag_B
        comparison = compare_tags(tag_A, tag_B)
        
        return jsonify({
            'success': True,
            'tag_a': tag_A,
            'tag_b': tag_B,
            'comparison': comparison
        })
    except Exception as e:
        logger.error(f"比较歌曲组时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)